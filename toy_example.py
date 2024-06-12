import cunumeric as np
import pdb
import time
import legate.core.types as ty
from legate.core import Store, legion
from legate.core._lib.context import ComposeClass
from legate.core._legion.geometry import Rect
from typing import cast

import pygion
from pygion import task, Fspace, Ispace, Region, RW, c, _my, index_launch


runtime = legion.legion_runtime_get_runtime()
context = legion.legion_runtime_get_context()

x = ComposeClass()
fillOp = x.registerFill()
printOp = x.registerPrint()
x.registerMapper(1)
mapperId = x.getID()
gpuTask = x.registerFake()


wrapped_field_spaces = {}

@task(privileges=[RW])
def init(R):
    R.x.fill(123)


# Reused fields do not have their existing names overwritten and the used name is returned
def create_pygion(array, field_name):
    field_id, logical_region, permission_owning_logical_region = array.extract_legion_resources()

    # Pygion assumes a field_name is given to each field in a field_space
    if logical_region.field_space.id not in wrapped_field_spaces:
        wrapped_field_spaces[logical_region.field_space.id] = field_name
        c.legion_field_id_attach_name(
            _my.ctx.runtime, logical_region.field_space, field_id, field_name.encode('utf-8'), False)
    field_name = wrapped_field_spaces[logical_region.field_space.id]

    region = Region.from_raw(logical_region)

    # An actual implementation would just connect legate dtypes to pygion types
    region.fspace.field_types[field_name] = pygion.int64

    return region, field_name




shape = (10000,1000)
rect = Rect(shape)

# Region represented in pure Legion
index_space = legion.legion_index_space_create_domain(runtime, context, rect.raw())
field_space = legion.legion_field_space_create(runtime, context)
field_allocator = legion.legion_field_allocator_create(runtime, context, field_space)
field_id = legion.legion_field_allocator_allocate_field(field_allocator, 8, 1000)
logical_region = legion.legion_logical_region_create(
    runtime,
    context,
    index_space,
    field_space,
    True,
)

# Legate representation
legate_rep = np.wrap_region_field(
    field_id,
    logical_region,
    logical_region,
    ty.int64
)

# Pygion representation
pygion_rep, field_name = create_pygion(legate_rep, 'x')

init(pygion_rep)

# Changes are propogated through the different worlds
print(legate_rep)

# Registering our own task with pygion
my_external_type = pygion.extern_task(
    task_id=gpuTask,
    argument_types=[Region],
    privileges=[RW('x')],
    return_type = pygion.void,
    calling_convention='regent')

my_external_type(pygion_rep)

print(legate_rep)