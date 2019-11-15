from abc import ABC, abstractmethod


class AbstractAxis(ABC):

    def __init__(self, name="", units=""):
        self.name = name
        self.units = units

    @property
    @abstractmethod
    def dimension(self):
        """

        :return:
        """

    @property
    @abstractmethod
    def volumes(self):
        """

        Returns:

        """

    @property
    @abstractmethod
    def coordinates(self):
        """

        Returns:

        """

    @property
    @abstractmethod
    def boundaries(self):
        """

        :return:
        """

    @property
    @abstractmethod
    def cell_edges(self):
        """

        :return:
        """

    @property
    @abstractmethod
    def size(self):
        """

        :return:
        """

    @property
    @abstractmethod
    def regular(self):
        """

        :return:
        """


class Abstract1dAxis(AbstractAxis):

    @property
    def dimension(self):
        return 1

    @abstractmethod
    def plot1d(self, data, *args, **kwargs):
        """

        :return:
        """

    # @abstractmethod
    # def to2d(self, axis2):
    #     """
    #
    #     :param axis2:
    #     :return:
    #     """


class Abstract2dAxis(AbstractAxis):

    @property
    def dimension(self):
        return 2

    @abstractmethod
    def plot2d(self, data, *args, **kwargs):
        """

        :return:
        """
