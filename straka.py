import torch
import math
import time
import kintera
import snapy
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput
        )

from torch.profiler import profile, record_function, ProfilerActivity

p0 = 1.0e5
Ts = 300.0
xc = 0.0
xr = 4.0e3
zc = 3.0e3
zr = 2.0e3
dT = -15.0
grav = 9.8
Rd = 287.0
gamma = 1.4
K = 75.0

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("straka.yaml");

# initialize block
block = MeshBlock(op)
block.to(dtype=torch.float64, device=torch.device("cpu"))

# get handles to modules
coord = block.hydro.module("coord")
eos = block.hydro.module("eos")
thermo = eos.named_modules()["thermo"]

# thermodynamics
Rd = kintera.constants.Rgas / kintera.species_weights()[0];
cv = kintera.species_cref_R()[0] * Rd;
cp = cv + Rd;

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

w = block.buffer("hydro.eos.W")

L = torch.sqrt(((x2v - xc) / xr) ** 2 + ((x1v - zc) / zr) ** 2)
temp = Ts - grav * x1v / cp

w[index.ipr] = p0 * torch.pow(temp / Ts, cp / Rd)
temp += torch.where(L <= 1, dT * (torch.cos(L * math.pi) + 1.0) / 2.0, 0)
w[index.idn] = w[index.ipr] / (Rd * temp)

block.initialize(w)

# make output
out2 = NetcdfOutput(OutputOptions().file_basename("straka").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("straka").fid(3).variable("uov"))
current_time = 0.0

block.set_uov("temp", temp)
block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))

for out in [out2, out3]:
    out.write_output_file(block, current_time)
    out.combine_blocks()

activities = [ProfilerActivity.CPU]

# integration
count = 0;
start_time = time.time()
interior = block.part((0, 0, 0))

# with profile(activities=activities, record_shapes=True) as prof:
while not block.intg.stop(count, current_time):
    dt = block.max_time_step()
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage)

    current_time += dt
    count += 1
    if count % 100 == 0:
        print("time = ", current_time)
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

print("elapsed time = ", time.time() - start_time)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
