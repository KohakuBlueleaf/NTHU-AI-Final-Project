from itertools import chain


def slicepart(l, block_size, slice_inside_block=slice(None, None)):
    return list(
        chain(
            *(
                l[i : i + block_size][slice_inside_block]
                for i in range(0, len(l), block_size)
            )
        )
    )
