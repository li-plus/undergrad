import threading


class SemReaderFirst(object):
    write_mutex = threading.Semaphore(1)
    count_mutex = threading.Semaphore(1)
    read_count = 0
    shared_data = 0

    @classmethod
    def read(cls, indent):
        with cls.count_mutex:
            print(indent + 'aCount')
            if cls.read_count == 0:
                cls.write_mutex.acquire()
            cls.read_count += 1
            print(indent + 'rCount')

        print(indent + 'rd ' + str(cls.shared_data))

        with cls.count_mutex:
            print(indent + 'aCount')
            cls.read_count -= 1
            if cls.read_count == 0:
                print(indent + 'rWrite')
                cls.write_mutex.release()
            print(indent + 'rCount')

    @classmethod
    def write(cls, indent):
        with cls.write_mutex:
            print(indent + 'aWrite')

            cls.shared_data += 1
            print(indent + 'wr ' + str(cls.shared_data))

            print(indent + 'rWrite')
