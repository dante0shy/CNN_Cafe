import os

descriptors = list()

def getDefaultSet():
    test_file = "nvidia-smi"

    try:
        p = os.popen(test_file, 'r')
        return p.read()

    except IOError:
        return "Error"

def getString():
    test_file = "nvidia-smi -q -d UTILIZATION"

    try:
        p = os.popen(test_file, 'r')
        return p.read()

    except IOError:
        return "Error"

## surport 9 or 10
def readl(key):
    output = getString()
    # print output
    splittedoutput = output.split('\n')
    i=0
    for line in splittedoutput:
        # line = line.strip()
        if(i==key):
            return float(line[38:40])
        i+=1



# Testing
if __name__ == '__main__':
    # metric_init({})
    # for d in descriptors:
    #     v = d['call_back'](d['name'])
    # print 'value for %s is %u' % (d['name'], v)\
    print getDefaultSet()
    print getString()
    print readl(9)