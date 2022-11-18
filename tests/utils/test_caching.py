from ml.utils.caching import Index


def test_index() -> None:
    col_dict = {k: list(range(k)) for k in range(1, 5)}
    ind = Index(col_dict)

    assert len(ind) == sum(range(1, 5))
    assert sum(ind[i][1] for i in range(len(ind))) == sum(i for j in range(1, 5) for i in range(j))
