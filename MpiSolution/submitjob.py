#!/usr/bin/python

import sys
import subprocess
import getopt
import random

# Create shell script to submit to qsub
# Results stored in file 'OUTROOT-XXXX.out' with specified number of digits
def generateScript(scriptName = "latedays.sh", argString = "", outputName = "benchmark.out"):
    try:
        scriptFile = open(scriptName, 'w')
    except Exception as e:
        print "Couldn't open file '%s' (%s)" % (scriptName, str(e))
        return False
    argString += " -f " + outputName
        
    scriptFile.write("#!/bin/bash\n")    
    scriptFile.write("# This script lets you submit jobs for execution on the latedays cluster\n")    
    scriptFile.write("# You should submit it using qsub:\n")    
    scriptFile.write("#   'qsub latedays.sh'\n")    
    scriptFile.write("# Upon completion, the output generated on stdout will show up in the\n")    
    scriptFile.write("# file latedays.sh.oNNNNN where NNNNN is the job number.  The output\n")    
    scriptFile.write("# generated on stderr will show up in the file latedays.sh.eNNNNN.\n")    
    scriptFile.write("\n")    
    # scriptFile.write("#PBS -q titanx\n") 
    scriptFile.write("# Limit execution time to 30 minutes\n")    
    scriptFile.write("#PBS -lwalltime=0:030:00\n")    
    scriptFile.write("# Allocate all available CPUs on a single node\n")    
    scriptFile.write("#PBS -l nodes=14:ppn=24\n")    
    scriptFile.write("\n")    
    scriptFile.write("# Go to the directory from which you submitted your job\n")
    scriptFile.write("cd $PBS_O_WORKDIR\n")    
    scriptFile.write("\n")    
    scriptFile.write("# Execute the performance evaluation program and store summary in %s\n" % outputName)
    scriptFile.write("mpirun -np 14 -bynode -bind-to-core --hostfile $PBS_NODEFILE ./inference-mpi \n")
    scriptFile.close()
    return True

def submit(scriptName):
    cmd = ["qsub", scriptName]
    cmdline = " ".join(cmd)
    try:
        process = subprocess.Popen(cmd)
    except Exception as e:
        print "Couldn't execute '%s' (%s)" % (cmdline, str(e))
        return
    process.wait()
    if process.returncode != 0:
        print "Error.  Executing '%s' gave return code %d" % (cmdline, process.returncode)

def run(name, args):
    scriptName = "latedays.sh"
    if generateScript():
        print "Generated script %s" % scriptName
        submit(scriptName)

if __name__ == "__main__":
    run(sys.argv[0], sys.argv[1:])
        
