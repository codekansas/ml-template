import ast
import functools
import inspect
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Set, Tuple, Type

from omegaconf import DictConfig

from ml.core.registry import (
    NAME_KEY,
    multi_register_base,
    register_base,
    register_logger,
    register_lr_scheduler,
    register_model,
    register_optimizer,
    register_task,
    register_trainer,
)
from ml.utils.call_train import call_train

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent.parent
BASE_NAME = __name__.split(".", maxsplit=1)[0]
BASE_MODULE_PREFIX = f"{BASE_NAME}."


def lookup_path(config: DictConfig, registry: Type[register_base]) -> Path:
    if registry.config_key() not in config:
        raise KeyError(f"Key '{registry.config_key()}' not found in config")
    if NAME_KEY not in config[registry.config_key()]:
        raise KeyError(f"'{NAME_KEY}' not found in {registry.config_key()} config")
    return registry.lookup_path(config[registry.config_key()][NAME_KEY])


def lookup_paths(config: DictConfig, registry: Type[multi_register_base]) -> List[Path]:
    if registry.config_key() not in config:
        raise KeyError(f"Key '{registry.config_key()}' not found in config")
    paths: List[Path] = []
    for subconfig in config[registry.config_key()]:
        if NAME_KEY not in subconfig:
            raise KeyError(f"'{NAME_KEY}' not found in {registry.config_key()} config")
        paths.append(registry.lookup_path(subconfig[NAME_KEY]))
    return paths


def get_loc_str(path: str | Path, node: ast.AST) -> str:
    return f"{path}:{node.lineno}:{node.col_offset}"


@dataclass(frozen=True)
class ImportItem:
    module: str
    name: str | None
    asname: str | None


def toposort(data: Dict[str, Set[str]]) -> Iterator[str]:
    if len(data) == 0:
        return
    data = {item: set(e for e in dep if e != item) for item, dep in data.items()}
    extra_items_in_deps = functools.reduce(set.union, data.values()) - set(data.keys())
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if len(dep) == 0)
        if not ordered:
            break
        yield from ordered
        data = {item: (dep - ordered) for item, dep in data.items() if item not in ordered}
    if data:
        raise RuntimeError("Found circular dependencies!")


@functools.lru_cache
def api_import_stmts() -> Dict[str, ast.ImportFrom]:
    api_path = ROOT_PATH / BASE_NAME / "api.py"

    with open(api_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    stmts: Dict[str, ast.ImportFrom] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            raise NotImplementedError("API should only have `from ... import ...` statements")
        if isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                key = alias.name if alias.asname is None else alias.asname
                stmts[key] = ast.ImportFrom(module=stmt.module, names=[alias], level=stmt.level)

    return stmts


def api_import_stmt(name: str, asname: str | None) -> ast.ImportFrom:
    stmt = api_import_stmts()[name]
    return ast.ImportFrom(module=stmt.module, names=[ast.alias(name=name, asname=asname)], level=stmt.level)


def replace_expr(base: ast.stmt, from_expr: ast.expr, to_expr: ast.expr) -> None:
    for field in base._fields:
        field_val = getattr(base, field)
        if field_val == from_expr:
            setattr(base, field, to_expr)
            break
        if isinstance(field_val, list):
            setattr(base, field, [to_expr if i == from_expr else i for i in field_val])
            break
    else:
        raise ValueError("Error while replacing expression")


def remove_api(api_name: str, tree: ast.AST) -> Set[ast.ImportFrom]:
    api_modules: Set[str] = set()

    todo: "Deque[ast.AST]" = deque([tree])
    while todo:
        node = todo.popleft()
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == api_name:
            new_node = ast.Name(id=node.attr)
            replace_expr(node.parent, node, new_node)
            api_modules.add(node.attr)
        for child in ast.iter_child_nodes(node):
            child.parent = node
            todo.append(child)

    return {api_import_stmt(c, None) for c in api_modules}


def preprocess_stmt(stmt: ast.stmt, tree: ast.AST) -> Iterator[ast.stmt]:
    if isinstance(stmt, ast.Import):
        for name in stmt.names:
            if name.name == f"{BASE_NAME}.api":
                yield from remove_api(name.name if name.asname is None else name.asname, tree)
            else:
                yield ast.Import(names=[name])
    elif isinstance(stmt, ast.ImportFrom):
        if stmt.module == BASE_NAME:
            for name in stmt.names:
                if name.name == "api":
                    yield from remove_api(name.name if name.asname is None else name.asname, tree)
        elif stmt.module == f"{BASE_NAME}.api":
            for name in stmt.names:
                yield api_import_stmt(name.name, name.asname)
        else:
            yield stmt
    else:
        yield stmt


def get_import_dag_for(
    path: Path,
    path_mod_names: Dict[Path, str],
    all_other_imports: Set[ImportItem],
    deps: Dict[str, Set[str]],
) -> None:
    mod_name = path_mod_names[path]
    if mod_name in deps:
        return
    deps[mod_name] = set()

    # Gets the module dag.
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(path))

    def process_stmt(stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Import):
            for name in stmt.names:
                if name.name == BASE_NAME or name.name.startswith(BASE_MODULE_PREFIX):
                    deps[mod_name].add(name.name)
                    mod_path = Path(inspect.getfile(sys.modules[name.name]))
                    get_import_dag_for(mod_path, path_mod_names, all_other_imports, deps)
                else:
                    all_other_imports.add(ImportItem(module=name.name, name=None, asname=name.asname))
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module is None or stmt.module.startswith("."):
                raise RuntimeError(f"Cannot compile program with relative imports: {get_loc_str(path, stmt)}")
            if stmt.module == BASE_NAME or stmt.module.startswith(BASE_MODULE_PREFIX):
                if stmt.module not in sys.modules:
                    raise RuntimeError(f"Module '{stmt.module}' has not been imported")
                deps[mod_name].add(stmt.module)
                mod_path = Path(inspect.getfile(sys.modules[stmt.module]))
                path_mod_names[mod_path] = stmt.module
                get_import_dag_for(mod_path, path_mod_names, all_other_imports, deps)
            else:
                for name in stmt.names:
                    all_other_imports.add(ImportItem(module=stmt.module, name=name.name, asname=name.asname))
        elif isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "TYPE_CHECKING":
            pass
        else:
            for i in ast.walk(stmt):
                if isinstance(i, (ast.ImportFrom, ast.Import)):
                    raise RuntimeError(f"Cannot compile program with non-top level imports: {get_loc_str(path, i)}")

    for stmt in (s for stmt in tree.body for s in preprocess_stmt(stmt, tree)):
        process_stmt(stmt)


def get_import_dag(paths: List[Path]) -> Tuple[Dict[str, Set[str]], Set[ImportItem]]:
    all_other_imports: Set[ImportItem] = set()
    deps: Dict[str, Set[str]] = {}
    path_mod_names = {Path(inspect.getfile(v)): k for k, v in sys.modules.items() if k.startswith(BASE_MODULE_PREFIX)}
    for path in paths:
        get_import_dag_for(path, path_mod_names, all_other_imports, deps)
    return deps, all_other_imports


def get_import_code_block(imports: Set[ImportItem]) -> str:
    """Builds a code block for the import statements.

    Args:
        imports: The set of all parsed import items

    Returns:
        A code block for the import statements
    """

    # Adds import statements.
    body: List[ast.stmt] = []
    import_names = (i for i in imports if i.name is None)
    for name, asname in sorted({(i.module, i.asname) for i in import_names}):
        body.append(ast.Import(names=[ast.alias(name=name, asname=asname)]))

    # Adds import from lines.
    import_from_map: Dict[str, Set[Tuple[str, str | None]]] = {i.module: set() for i in imports if i.name is not None}
    for i in imports:
        if i.name is not None:
            import_from_map[i.module].add((i.name, i.asname))
    for mod_name, items in sorted(import_from_map.items()):
        if len(items) == 1:
            name, asname = list(items)[0]
            body.append(ast.ImportFrom(module=mod_name, names=[ast.alias(name=name, asname=asname)], level=0))
        else:
            import_from_node = ast.ImportFrom(module=mod_name, names=[], level=0)
            for name, asname in items:
                import_from_node.names.append(ast.alias(name=name, asname=asname))
            body.append(import_from_node)

    base = ast.Module(body=body, type_ignores=[])
    return ast.unparse(base)


def get_file_block(mod_path: Path) -> str:
    with open(mod_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(mod_path))

    new_stmts: List[ast.stmt] = []

    for stmt in (s for stmt in tree.body for s in preprocess_stmt(stmt, tree)):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        elif isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "TYPE_CHECKING":
            continue
        else:
            new_stmts.append(stmt)

    return ast.unparse(ast.Module(body=new_stmts, type_ignores=[]))


def make_title(title: str) -> str:
    return "\n".join(["#" * (len(title) + 4), f"# {title} #", "#" * (len(title) + 4)])


def get_full_file(deps: Dict[str, Set[str]], imports: Set[ImportItem]) -> str:
    header = "#!/usr/bin/env python\n\n"

    parts: List[Tuple[str, str]] = []
    parts.append(("imports", get_import_code_block(imports)))

    # Iterates through dependencies in topographical order.
    for dep in toposort(deps):
        mod_path = Path(inspect.getfile(sys.modules[dep]))
        parts.append((str(mod_path.relative_to(ROOT_PATH)), get_file_block(mod_path)))

    return header + "\n\n".join(f"{make_title(title)}\n\n{body}" for title, body in parts)


def compile_training_loop(config: DictConfig, out_path: Path) -> None:
    """Compiles the full training loop to a single file.

    Args:
        config: The training config
        out_path: The path to the output file
    """

    model_path = lookup_path(config, register_model)
    trainer_path = lookup_path(config, register_trainer)
    task_path = lookup_path(config, register_task)
    optimizer_path = lookup_path(config, register_optimizer)
    lr_scheduler_path = lookup_path(config, register_lr_scheduler)
    logger_paths = lookup_paths(config, register_logger)
    call_train_path = Path(inspect.getfile(call_train))
    call_train_mod = f"{BASE_NAME}.utils.call_train"

    paths = logger_paths + [optimizer_path, lr_scheduler_path, trainer_path, task_path, model_path, call_train_path]
    deps, all_other_imports = get_import_dag(paths)
    deps[call_train_mod].update(i for i in deps if i != call_train_mod)
    full_file = get_full_file(deps, all_other_imports)

    # If black is installed, format the raw file contents.
    try:
        import black  # pylint: disable=import-outside-toplevel

        full_file = black.format_file_contents(full_file, fast=True, mode=black.FileMode())
    except ModuleNotFoundError:
        pass

    out_path.parent.mkdir(exist_ok=True, parents=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_file)
