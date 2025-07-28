import torch
import math
import time
import numpy as np
from snapy import (
        MeshBlockOptions,
        MeshBlock
        )
from kintera import ThermoX

torch.set_default_dtype(torch.float64)

# parameters
Ts = 300.
Ps = 1.e5
Tmin = 110.
grav = 9.8
xH2O = 0.02

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("earth.yaml");
print(op)

# initialize block
block = MeshBlock(op)
block.to(torch.device("cuda:0"))

# get handles to modules
coord = block.hydro.module("coord")
eos = block.hydro.module("eos")
thermo_y = eos.named_modules()["thermo"]

# get coordinates
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

# get primitive variable
w = block.buffer("hydro.eos.W")

# get dimensions
nc3, nc2, nc1 = x1v.shape
ny = len(thermo_y.options.species()) - 1

temp = Ts * torch.ones(nc3, nc2)
pres = Ps * torch.ones(nc3, nc2)
xfrac = torch.ones(nc3, nc2, 1)
xfrac /= xfrac.sum(-1, keepdim=True)
print(xfrac)

# half a grid to cell center
dz = 0.;

thermo_x = ThermoX(thermo_y.options)
thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz);


temp = Ts - grav * x1v / cp
w[index.ipr] = p0 * torch.pow(temp / Ts, cp / Rd)
tempv = temp.clone()

L = torch.sqrt(((x2v - xc) / xr)**2 + ((x1v - zc) / zr)**2)
tempv *= torch.where(L <= 1, 1. + dT * cos(np.pi * L / 2.)**2 / 300., 1.0)

w[index.idn] = w[index.ipr] / (Rd * temp)

block.initialize(w)

# make output
# out1 = AsciiOutput(OutputOptions().file_basename("robert").fid(1).variable("hst"))
out2 = NetcdfOutput(OutputOptions().file_basename("robert").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("robert").fid(3).variable("uov"))
current_time = 0.0

block.set_uov("temp", temp)
block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))
block.set_entropy("entropy", w[index.ipr], w[index.idn])

exit()

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
