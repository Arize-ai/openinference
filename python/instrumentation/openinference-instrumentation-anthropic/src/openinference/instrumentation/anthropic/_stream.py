from wrapt import ObjectProxy


class _Stream(ObjectProxy):
    def __init__(
            self,
            stream,
    ) -> None:
        super().__init__(stream)

    def __iter__(self):
        for item in self.__wrapped__:
            print("harrison got {}".format(item))
            yield item
