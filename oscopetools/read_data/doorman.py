"""Classes for providing temporary mutability."""

from abc import ABC, abstractmethod

__all__ = ('Unlockable', 'LockError')


class Unlockable(ABC):
    """Base class for objects that can be temporarily unlocked.

    Unlockable objects are usually immutable, but can be made temporarily
    mutable by calling the unlock() method. This is useful to prevent
    accidental modification of objects containing data that generally should
    not be changed, while still allowing the possibility of modification if
    needed.

    Methods
    -------
    unlock()

    Notes
    -----
    Subclasses must implement reversible locking behaviour by defining
    `_unlock()` and `_lock()`.

    """

    def unlock(self):
        """Temporarily unlock self.

        Returns
        -------
        Doorman
            Context manager than unlocks self on entry and re-locks it on exit.

        Usage
        -----
        >>> with unlockable_object.unlock():
        >>>     # Do something that requires mutability
        >>>     pass

        """
        return Doorman(self)

    @abstractmethod
    def _unlock(self):
        """Temporarily unlock self."""
        # Make some immutable attributes of self temporarily mutable
        raise NotImplementedError

    @abstractmethod
    def _lock(self):
        """Lock self to prevent accidental modification."""
        # Reverse the effects of _unlock()
        raise NotImplementedError


class Doorman:
    """Context manager for temporarily unlocking objects."""

    __slots__ = ('__room')

    def __init__(self, room: Unlockable):
        """Initialize Doorman with an object to lock/unlock.

        Parameter
        ---------
        room : Unlockable
            Object to unlock when the context manager is entered, and re-lock
            when the context manager exits.

        """
        if not issubclass(type(room), Unlockable):
            raise TypeError(
                'Expected argument `room` to be a subclass of `Unlockable`, '
                'got instance of type {} instead.'.format(type(room))
            )

        self.__room = room

    def __enter__(self):
        """Unlock the room so that the user may enter."""
        self.__room._unlock()

    def __exit__(self, err_type, err_val, err_tb):
        """Lock the room behind the user on exit."""
        self.__room._lock()


class LockError(Exception):
    pass
