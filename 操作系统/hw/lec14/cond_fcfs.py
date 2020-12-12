import threading


class CondFCFS(object):
    ref_lock = threading.Condition()
    ref_count = 0
    shared_data = 0

    @classmethod
    def read(cls, indent):
        with cls.ref_lock:
            print(indent + 'aLock')
            while cls.ref_count > 0:
                print(indent + 'wait')
                cls.ref_lock.wait()
            cls.ref_count += 1
            print(indent + 'rLock')

        print(indent + 'rd ' + str(cls.shared_data))

        with cls.ref_lock:
            print(indent + 'aLock')
            cls.ref_count -= 1
            cls.ref_lock.notify()
            print(indent + 'rLock')

    @classmethod
    def write(cls, indent):
        with cls.ref_lock:
            print(indent + 'aLock')
            while cls.ref_count > 0:
                print(indent + 'wait')
                cls.ref_lock.wait()
            cls.ref_count += 1
            print(indent + 'rLock')

        cls.shared_data += 1
        print(indent + 'wr ' + str(cls.shared_data))

        with cls.ref_lock:
            print(indent + 'aLock')
            cls.ref_count -= 1
            cls.ref_lock.notify()
            print(indent + 'rLock')
