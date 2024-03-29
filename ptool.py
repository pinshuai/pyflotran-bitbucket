"""Tools for use with pflotran python"""

import numpy as np
import os
import platform
import string
from pdflt import *
import warnings

WINDOWS = platform.system() == 'Windows'
if WINDOWS:
    slash = '\\'
else:
    slash = '/'

FULL_PRECISION = False

dflt = pdflt()

def decode(byteseq):
    if isinstance(byteseq,bytes):
        try:
            return byteseq.decode('ascii')
        except UnicodeDecodeError as e:
            print(byteseq)
            print('COULD NOT DECODE: ')
            print(byteseq[e.start:e.end])
            raise UnicodeDecodeError(e.encoding,e.object,e.start,e.end,e.reason)
    else:
        return byteseq

class Frozen(object):
    """
    Prevents adding new attributes to classes once _freeze() is
    called on the class.
    """
    __frozen = False

    def __setattr__(self, key, value):
        if not self.__frozen or hasattr(self, key):
            object.__setattr__(self, key, value)
        else:
            raise AttributeError(
                str(key) + ' is not a valid attribute for ' +
                self.__class__.__name__)

    def _freeze(self):
        self.__frozen = True

    def _unfreeze(self):
        self.__frozen = False


def powspace(x0, x1, N=10, power=1):
    """
    Returns a sequence of numbers spaced according to the power law
    (x1-x0)**(1-power)*linspace(0,(x1-x0),N)**base + x0

    :param x0: First number in sequence.
    :type x0: fl64
    :param x1: Last number in sequence.
    :type x1: fl64
    :param N: Total items in sequence.
    :type N: int
    :param power: Index of power law. If negative, spacing order will be
     reversed from "big-to-small".
    :type power: fl64
    """
    if power > 0:
        return (x1 - x0) ** (1 - power) * np.linspace(0, x1 - x0, N) ** \
            power + x0
    elif power < 0:
        return np.sort(x1 - ((x1 - x0) ** (1 - abs(power)) * \
                       np.linspace(0, x1 - x0, N) ** abs(power)))


def del_extra_slash(path):
    return path.replace(slash + slash, slash)


def floatD(string):
    """Converts input number string to float, replacing 'd' with 'e'

    :param string: Number in string format.
    :type string: str
    """
    if not isinstance(string, str):
        string = strD(string)  # Convert to string if not string.
    if '*' in string:
        return eval(string.replace('d','e'))
    return float(string.lower().replace('d', 'e'))

def boolS(string):
    """
    Converts an input string to a Boolean.
    Evaluates against ['1','true','t','yes','y','on'].

    :param string: String to be converted
    :type string: string
    """
    return string.strip().lower in ['1','true','t','yes','y','on']

def strD(number):
    """Converts input float to string, replacing 'e' with 'd'

    :param number: Number to be converted.
    :type number: float
    """
    tstring = str(number)
    if 'e' in tstring:
        width = len(tstring[:tstring.find('e')])
        if '.' in tstring:
            width -= len(tstring[:tstring.find('.')])
        return ('%.*e' % (width,number)).replace('e', 'd')
    else:
        return tstring


def strI(number):
    """Converts input integer to string

    :param number: Number to be converted.
    :type number: int
    """
    tstring = str(int(number))
    return tstring

def strB(logical):
    """
    Converts a Boolean into a string.
    """
    return str(logical).upper()

def line_from_substr(line,substr='DBASE_VALUE',at_front=True):
    """
    Returns the substring of a line beginning at
    substr and ending at EOL.

    Example:
    >>> line_from_substr('ID DBASE_VALUE soil2_id')
    DBASE_VALUE soil2_id

    :param line: string to parse
    :type line: str
    :param substr: substring to start capture at
    :type substr: str
    :param at_front: Flag to indicate if capture should begin
    at the beginning of substr or at the end
    :type at_front: bool
    """

    if at_front is True:
        return line[line.lower().find(substr.lower()):]
    else:
        return line[line.lower().find(substr.lower()) + 
        len(substr):]

def is_comment(line):
    """
    Returns a true or false value if the line is a commented line
    Example:

    >>> is_comment('# TODO: add is_comment func')
    True
    >>> is_comment('NEWTON_SOLVER TRANSPORT')
    False

    :param line: string to test
    :type line: str
    """

    return line.strip()[0] in ['#','!']

def get_next_line(infile):
    """
    Returns the next valid line in a file.
    That is, no (i) commented lines are
    returned, (ii) no items within a skip 
    block are returned, (iii) no empty lines
    are returned, and (iv) any valid lines with
    trailing comments have those comments removed.

    example_file.in:
        # Comment 1
        ! Comment 2
        FIRST_LINE # with comment
        # Comment 3
        SECOND_LINE

    >>> infile = open('example_file.in','r')
    >>> get_next_line(infile)
    FIRST_LINE
    >>> get_next_line(infile)
    SECOND_LINE

    :param infile: Open file-read stream
    :type infile: file stream (_io.TextIOWrapper)
    """

    while True:
        line = infile.readline()

        # Check for SKIP/NOSKIP
        if line.strip().lower() == 'skip':
            while True:
                pline = infile.readline()
                if pline.strip().lower() == 'noskip':
                    break
        else:
            # Check for empty lines
            if line.strip() == '': continue

            # Check for extended line characters: \
            if "\\" in filter_comment(line):
                extended_line = ""

                while True:
                    idx = line.find("\\")

                    # If there are no more continuation chars,
                    # then append current line and quit
                    if idx == -1:
                        extended_line += line
                        return extended_line.strip()

                    # Get line up to \
                    extended_line += line[:idx]
                    line = filter_comment(infile.readline())

            # Check for comments
            if not line.strip()[0] in ['#','!']:
                return filter_comment(line)

def get_key(line):
    """
    Returns the card key used in file parsing.
    """
    return line.strip().split()[0].lower()

def strD_matrix(data,columns=5,width=15,indent=0):
    """
    Converts a 1D vector or list to a string matrix of a defined
    amount of columns.

    Useful for writing to DATA blocks.
    """

    if len(data) <= columns:
        return ' '.join([strD(x).ljust(width) for x in data]) + '\n'

    s = ''
    for (i,v) in enumerate(data,1):
        s += strD(v).ljust(width) + ' '
        if not i % columns:
            s += '\\ \n'
    return ' '*indent + s[:-2].replace('\n','\n%s' % (' '*indent)) + '\n'

def filter_comment(line):
    """
    Returns a line with any trailing comment removed.
    Example:

        >>> filter_comment('8 ! west')
        8

    :param line: string to filter
    :type line: str
    """
    
    comments = ['!','#']

    for cchar in comments:
        idx = line.find(cchar)
        if idx > -1:
            return line[:idx].strip()

    return line


def PyFLOTRAN_WARNING(string):
    return warnings.warn(string)


class PyFLOTRAN_ERROR(Exception):
    pass


def perror(string):
    """Prints out an error

    :param string: Error string
    :type string: str
    """
    raise PyFLOTRAN_ERROR(string)


def pyflotran_print(s):
    if not pyflotran_print.silent:
        print(s)


# -- FUNCTIONS AND CLASSES FOR INTERNAL USE --

class ppath(object):

    def __init__(self, filename='', work_dir='', parent=None):
        self._filename = filename
        self.absolute_to_file = ''
        # location where originally read DOES NOT CHANGE
        self.absolute_to_workdir = ''
        # working directory CAN CHANGE
        self.parent = parent

    def update(self, wd):
        """called when work_dir is updated"""
        if wd == '':
            self.absolute_to_workdir = ''
            return

        if WINDOWS:
            wd = wd.replace('/', '\\')
        else:
            wd = wd.replace('\\', '/')

        absolute = False
        if WINDOWS and wd[1] == ':':
            absolute = True
        if not WINDOWS and wd[0] == '/':
            absolute = True
        if absolute:
            self.absolute_to_workdir = wd
        else:
            self.absolute_to_workdir = os.getcwd() + slash + wd

    def _get_filename(self):
        return self._filename

    def _set_filename(self, value):
        # ensure path specification consistent with OS
        if WINDOWS:
            value = value.replace('/', '\\')
        else:
            value = value.replace('\\', '/')
        # check if any slashes exist
        if slash in value:
            self._filename = value.split(slash)[-1]
        else:
            self._filename = value
            self.absolute_to_file = os.getcwd()
            return
        # check if absoulte or relative specification
        path = value.split(slash)[:-1]
        path = slash.join(path)

        absolute = False
        if WINDOWS and path[1] == ':':
            absolute = True
        if not WINDOWS and path[0] == '/':
            absolute = True

        if absolute:
            self.absolute_to_file = path
        else:
            self.absolute_to_file = os.getcwd() + slash + path

    filename = property(_get_filename, _set_filename)  #: (**)

    def _get_full_path(self):
        return self.absolute_to_file + slash + self.filename

    full_path = property(_get_full_path)  #: (**)


def os_path(path):
    if WINDOWS:
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
    return path


# returns file name and extension for saving
def save_name(save='', variable='', time=0., node=0):
    if save:
        save = save.split('.')
        if len(save) == 1:
            print('No extension specified, default to .png')
            ext = 'png'
        elif len(save) > 2:
            print('Too many dots!')
            return
        else:
            if save[1] in ['png', 'eps', 'pdf']:
                ext = save[1]
            else:
                print('Unrecognized extension')
                return
        save_fname = save[0] + '.' + ext
    else:
        from glob import glob

        ext = 'png'
        if time:
            varStr = variable + '_time' + str(time)
        elif node:
            varStr = variable + '_node' + str(node)
        else:
            varStr = variable
        files = glob('pyflotran_sliceplot_' + varStr + '_*.png')
        if not files:
            ind = 1
        else:
            inds = []
            for file in files:
                file = file.split('pyflotran_sliceplot_' + varStr + '_')
                inds.append(int(file[1].split('.png')[0]))
            ind = np.max(inds) + 1
        save_fname = "pyflotran_sliceplot_" + varStr + '_' + str(ind) + '.png'
    return ext, save_fname
