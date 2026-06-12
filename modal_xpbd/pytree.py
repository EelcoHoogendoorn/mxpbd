"""pytree registration of dataclasses

Fields marked with metadata static=True become aux data
(must be hashable; compared by equality under jit),
all other fields are traced array leaves.
"""
import dataclasses

import jax


def static_field(**kwargs):
	"""dataclass field treated as static aux data rather than a traced leaf"""
	return dataclasses.field(metadata=dict(static=True), **kwargs)


def register(cls):
	fields = dataclasses.fields(cls)
	data = [f.name for f in fields if not f.metadata.get('static', False)]
	meta = [f.name for f in fields if f.metadata.get('static', False)]
	return jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
