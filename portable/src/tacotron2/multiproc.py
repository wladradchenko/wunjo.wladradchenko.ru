"""
BSD 3-Clause License

Copyright (c) 2018, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import time
import sys
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("script", nargs="?")
parser.add_argument("-p", "--hparams_path", type=str, default="./data/hparams.yaml",
                    required=False, help="hparams path")
parser.add_argument("--gpus_ranks", type=str, default="",
                    required=False, help="gpu's indices for distributed run (separated by commas)")
parser.add_argument("--logs_path", type=str, default="data/logs",
                    required=False, help="path to logs")

args = parser.parse_args()

logs_path = os.path.abspath(args.logs_path)
os.makedirs(logs_path, exist_ok=True)

argslist = list(sys.argv)[1:2]
gpus_ranks = args.gpus_ranks.split(",")
workers = []

job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append("--hparams_path={}".format(args.hparams_path))
argslist.append("--gpus_ranks={}".format(args.gpus_ranks))
argslist.append("--group_name=group_{}".format(job_id))
argslist.append("-d")

for i, idx in enumerate(gpus_ranks):
    argslist.append("--gpu_idx={}".format(idx))
    stdout = None if i == 0 else open(os.path.join(logs_path, "{}_GPU_{}.log".format(job_id, i)), "w")

    print(argslist)
    p = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

for p in workers:
    p.wait()
