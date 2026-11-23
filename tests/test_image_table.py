import pytest
from Orange.data import Table

from orangecontrib.imagenets.util.image_table import image_table_variables


def test_finds_image_column(image_table, tmp_path):
    origin, index = image_table_variables(image_table)
    assert origin == str(tmp_path)
    assert image_table.domain.metas[index].name == "image"


def test_raises_without_image_column():
    with pytest.raises(Exception, match="image"):
        image_table_variables(Table("iris"))
