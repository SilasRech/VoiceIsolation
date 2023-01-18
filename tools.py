import sys


def splitStrings(st, dl):
    word = ""
    # to count the number of split strings
    num = 0
    # adding delimiter character at
    # the end of 'str'
    st += dl
    # length of 'str'
    l = len(st)
    # traversing 'str' from left to right
    substr_list = []
    for i in range(l):
        # if str[i] is not equal to the
        # delimiter character then accumulate
        # it to 'word'
        if (st[i] != dl):
            word += st[i]
        else:

            # if 'word' is not an empty string,
            # then add this 'word' to the array
            # 'substr_list[]'
            if (len(word) != 0):
                substr_list.append(word)
            # reset 'word'
            word = ""
    # return the splitted strings
    return substr_list


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        sys.stdout.write('\r')
        sys.stdout.write('\t'.join(entries))
        sys.stdout.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'