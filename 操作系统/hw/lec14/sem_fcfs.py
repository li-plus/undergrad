import threading


class SemFCFS(object):
    access_mutex = threading.Semaphore(1)
    reader_mutex = threading.Semaphore(1)
    order_mutex = threading.Semaphore(1)
    read_count = 0
    shared_data = 0

    @classmethod
    def read(cls, indent):
        with cls.order_mutex:
            print(indent + 'aOrder')
            with cls.reader_mutex:
                print(indent + 'aCount')
                if cls.read_count == 0:
                    cls.access_mutex.acquire()
                    print(indent + 'aAccess')
                cls.read_count += 1
                print(indent + 'rCount')
            print(indent + 'rOrder')

        print(indent + 'rd ' + str(cls.shared_data))

        with cls.reader_mutex:
            print(indent + 'aCount')
            cls.read_count -= 1
            if cls.read_count == 0:
                print(indent + 'rAccess')
                cls.access_mutex.release()
            print(indent + 'rCount')

    @classmethod
    def write(cls, indent):
        with cls.order_mutex:
            print(indent + 'aOrder')
            cls.access_mutex.acquire()
            print(indent + 'aAccess')
            print(indent + 'rOrder')

        cls.shared_data += 1
        print(indent + 'wr ' + str(cls.shared_data))

        print(indent + 'rAccess')
        cls.access_mutex.release()
