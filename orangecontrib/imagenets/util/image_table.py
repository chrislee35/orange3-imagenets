from Orange.data import Table

def image_table_variables(data: Table) -> [str, int]:
    domain = data.domain
    image_col = None
    origin = None
    for var in domain.metas:
        if var.attributes.get("type") == "image":
            image_col = var
            origin = var.attributes.get("origin")
            break
    if image_col is None:
        raise Exception("No variable with type \"image\"")
    image_col_index = domain.metas.index(image_col)
    return origin, image_col_index
