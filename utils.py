import time
class ElapsedTimer(object):
    def __init__(self,string='Elapsed'):
        self.start_time = time.time()
        self.string = string

    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print(self.string + ": %s" % self.elapsed(time.time() - self.start_time),
              flush=True)
        return (self.string + ": %s" % self.elapsed(time.time() - self.start_time))
