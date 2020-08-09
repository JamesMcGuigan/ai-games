#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from subprocess import PIPE
from typing import List



parser = argparse.ArgumentParser(
    description='Compile a list of python files into a Kaggle compatable script: \n' +
                './kaggle_compile.py [script_files.py] --save'
)
parser.add_argument('files', nargs='+',                                help='list of files to parse' )
parser.add_argument('--python-path', default='.',                      help='directory to search for local namespace imports')
parser.add_argument('--output-dir',  default='./',                     help='directory to write output if --save')
parser.add_argument('--save',        action='store_true',              help='should file be saved to disk')
parser.add_argument('--commit',      action='store_true',              help='should saved file be commited to git')
args, unknown = parser.parse_known_args()  # Ignore extra CLI args passed in by Kaggle
if len(args.files) == 0:  parser.print_help(sys.stderr); sys.exit();


module_names = [ name for name in os.listdir(args.python_path)
                 if os.path.isdir(os.path.join(args.python_path, name))
                 and not name.startswith('.') ]
module_regex = '(?:' + "|".join(map(re.escape, module_names)) + ')'
import_regex = re.compile(fr'^from\s+({module_regex}\b.*?)\s+import', re.MULTILINE)
assert_regex = re.compile(fr'\bassert\s', re.MULTILINE)


def read_and_comment_file(filename: str) -> str:
    code = open(filename, 'r').read()
    code = re.sub(import_regex, r'# \g<0>', code)  # comment out from * import statements
    code = re.sub(assert_regex, r'# \g<0>', code)  # comment out asserts for production
    return code


# TODO: handle "import src.module" syntax
# TODO: handle "from src.module.__init__.py import" syntax
def extract_dependencies_from_file(filename: str) -> List[str]:
    code    = open(filename, 'r').read()
    imports = re.findall(import_regex, code)
    files   = list(map(lambda string: string.replace('.', '/')+'.py', imports))
    return files


def recurse_dependencies(filelist: List[str]) -> List[str]:
    output = filelist
    for filename in filelist:
        dependencies = extract_dependencies_from_file(filename)
        if len(dependencies):
            output = [
                recurse_dependencies(dependencies),
                dependencies,
                output
            ]
    output = flatten(output)
    return output


def flatten(filelist):
    output = []
    for item in filelist:
        if isinstance(item,list):
            if len(item):         output.extend(flatten(item))
        else:                     output.append(item)
    return output


def unique(filelist: List[str]) -> List[str]:
    seen   = {}
    output = []
    for filename in filelist:
        if not seen.get(filename, False):
            seen[filename] = True
            output.append(filename)
    return output


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def savefile():
    savefile = os.path.join( args.output_dir, os.path.basename(args.files[-1]) )  # Assume last provided filename
    return savefile


def compile_script(filelist: List[str]) -> str:
    filelist = unique(filelist)


    shebang = "#!/usr/bin/env python3"
    header = [
        ("\n" + (" ".join(sys.argv)) + "\n"),
        subprocess.run('date --rfc-3339 seconds',     shell=True, stdout=PIPE).stdout.decode("utf-8"),
        subprocess.run('git remote -v',               shell=True, stdout=PIPE).stdout.decode("utf-8"),
        subprocess.run('git branch -v ',              shell=True, stdout=PIPE).stdout.decode("utf-8"),
        subprocess.run('git rev-parse --verify HEAD', shell=True, stdout=PIPE).stdout.decode("utf-8"),
    ]
    if args.save: header += [ f'Wrote: {savefile()}' ]

    header = map(lambda string: string.split("\n"), header )
    header = map(lambda string: '##### ' + string, flatten(header))
    header = "\n".join(flatten(header))

    output_lines = [
        shebang,
        header,
    ]
    for filename in filelist:
        output_lines += [
            f'#####\n##### START {filename}\n#####',
            read_and_comment_file(filename),
            f'#####\n##### END   {filename}\n#####',
        ]
    output_lines += [ header ]
    output_text   = "\n\n".join(output_lines)
    output_text   = reorder_from_future_imports(output_text)
    return output_text

def reorder_from_future_imports(output_text):
    lines   = output_text.split('\n')
    shebangs = [ line for line in lines[:10] if line.startswith('#!/')         ]
    futures  = [ line for line in lines      if '__future__' in line           ]
    other    = [ line for line in lines      if line not in shebangs + futures ]
    output   = "\n".join([ *shebangs, *sorted(set(futures)), *other ])
    return output



if __name__ == '__main__':
    filenames = recurse_dependencies(args.files)
    code      = compile_script(filenames)
    print(code)
    if args.save or args.commit:
        with open(savefile(), 'w') as file:
            file.write(code)
            file.close()
        make_executable(savefile())

        if args.commit:
            while not os.path.exists(savefile()): continue
            command = f'git add {savefile()}; git commit -o {savefile()} -m "kaggle_compile.py | {savefile()}"'
            print(f'$ {command}')
            print( subprocess.check_output(command, shell=True).decode("utf-8") )
