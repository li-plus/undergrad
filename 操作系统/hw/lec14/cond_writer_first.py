import threading


class CondWriterFirst(object):
    lock = threading.Lock()

    read_lock = threading.Condition(lock)
    write_lock = threading.Condition(lock)

    active_writers = 0
    active_readers = 0
    waiting_writers = 0
    waiting_readers = 0

    shared_data = 0

    @classmethod
    def read(cls, indent):
        with cls.lock:
            print(indent + 'aLock')
            while cls.active_writers > 0 or cls.waiting_writers > 0:
                cls.waiting_readers += 1
                print(indent + 'wait')
                cls.read_lock.wait()
                cls.waiting_readers -= 1
            cls.active_readers += 1
            print(indent + 'rLock')

        print(indent + 'rd ' + str(cls.shared_data))

        with cls.lock:
            print(indent + 'aLock')
            cls.active_readers -= 1
            if cls.active_readers == 0 and cls.waiting_writers > 0:
                cls.write_lock.notify()
            print(indent + 'rLock')

    @classmethod
    def write(cls, indent):
        with cls.lock:
            print(indent + 'aLock')
            while cls.active_readers > 0 or cls.active_writers > 0:
                cls.waiting_writers += 1
                print(indent + 'wait')
                cls.write_lock.wait()
                cls.waiting_writers -= 1

            cls.active_writers += 1
            print(indent + 'rLock')

        cls.shared_data += 1
        print(indent + 'wr ' + str(cls.shared_data))

        with cls.lock:
            print(indent + 'aLock')
            cls.active_writers -= 1

            if cls.waiting_writers > 0:
                cls.write_lock.notify()
            elif cls.waiting_readers > 0:
                cls.read_lock.notifyAll()
            print(indent + 'rLock')
