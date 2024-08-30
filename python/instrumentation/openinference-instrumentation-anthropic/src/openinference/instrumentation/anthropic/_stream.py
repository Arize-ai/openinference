from wrapt import ObjectProxy


class _Stream(ObjectProxy):
    def __init__(
            self,
            stream,
    ) -> None:
        super().__init__(stream)

    def __iter__(self):
        completion = self.__wrapped__.__iter__()
        print("harrison got {}".format(completion))
        yield completion