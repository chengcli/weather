import torch
import math
import time
from snapy import *

dT = 0.5
p0 = 1.0e5
Ts = 303.15
xc = 500.0
yc = 0.0
zc = 260.0
s = 100.0
a = 50.0
grav = 9.8
Rd = 287.0
gamma = 1.4
uniform_bubble = False

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("robert.yaml");

# initialize block
block = MeshBlock(op)
block.to(torch.device("cuda:0"))

# get handles to modules
coord = block.hydro.module("coord")
eos = block.hydro.module("eos")
thermo = eos.named_modules()["thermo"]

# thermodynamics
cp = gamma / (gamma - 1.0) * Rd

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

w = block.buffer("hydro.eos.W")

temp = Ts - grav * x1v / cp
w[index.ipr] = p0 * torch.pow(temp / Ts, cp / Rd)

r = torch.sqrt((x3v - yc) ** 2 + (x2v - xc) ** 2 + (x1v - zc) ** 2)
temp += torch.where(r <= a, dT * torch.pow(w[index.ipr] / p0, Rd / cp), 0.0)
if not uniform_bubble:
    temp += torch.where(
        r > a,
        dT * torch.exp(-(((r - a) / s) ** 2)) * torch.pow(w[index.ipr] / p0, Rd / cp),
        0.0,
    )
w[index.idn] = w[index.ipr] / (Rd * temp)

block.initialize(w)

# make output
# out1 = AsciiOutput(OutputOptions().file_basename("robert").fid(1).variable("hst"))
out2 = NetcdfOutput(OutputOptions().file_basename("robert").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("robert").fid(3).variable("uov"))
current_time = 0.0

block.set_uov("temp", temp)
block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))

for out in [out2, out3]:
    out.write_output_file(block, current_time)
    out.combine_blocks()

# integration
count = 0
start_time = time.time()
interior = block.part((0, 0, 0))
dt_max = 0.

while not block.intg.stop(count, current_time):
    dt = block.max_time_step()
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage)
    dt_max = max(dt_max, dt)

    current_time += dt
    count += 1
    if count % 1000 == 0:
        print("time = ", current_time)
        print("dt_max = ", dt_max)
        u = block.buffer("hydro.eos.U")
        print("mass = ", u[interior][index.idn].sum())

        ivol = thermo.compute("DY->V", (w[index.idn], w[index.icy:]))
        temp = thermo.compute("PV->T", (w[index.ipr], ivol))

        block.set_uov("temp", temp)
        block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))

        for out in [out2, out3]:
            out.increment_file_number()
            out.write_output_file(block, current_time)
            out.combine_blocks()

print("dt_max = ", dt_max)
print("elapsed time = ", time.time() - start_time)
