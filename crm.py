import torch
import math
import time
import kintera
import numpy as np
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput,
        )
from kintera import (
        ThermoOptions,
        ThermoX,
        KineticsOptions,
        Kinetics,
        )

torch.set_default_dtype(torch.float64)

def evolve_kinetics(block, kinet, thermo_x):
    eos = block.module("hydro.eos")
    thermo_y = eos.named_modules()["thermo"]

    w = block.buffer("hydro.eos.W")

    temp = eos.compute("W->T", (w,))
    pres = w[index.ipr]
    xfrac = thermo_y.compute("Y->X", (w[ICY:],))
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))
    cp_vol = thermo_x.compute("TV->cp", (temp, conc))

    conc_kinet = kinet.options.narrow_copy(conc, thermo_y.options)
    rate, rc_ddC, rc_ddT = kinet.forward(temp, pres, conc_kinet)
    jac = kinet.jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT)

    stoich = kinet.buffer("stoich")
    del_conc = kintera.evolve_implicit(rate, stoich, jac, dt)

    inv_mu = thermo_y.buffer("inv_mu")
    del_rho = del_conc / inv_mu[1:].view((1, 1, 1, -1))
    return del_rho.permute((3, 0, 1, 2))

def setup_initial_condition(block, thermo_x):
    Ts = 300.
    Ps = 1.e5
    xH2O = 0.02
    Tmin = 110.
    grav = 9.8

    # get handles to modules
    coord = block.module("hydro.coord")
    thermo_y = block.module("hydro.eos.thermo")

    # get coordinates
    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    # get dimensions
    nc3, nc2, nc1 = x1v.shape
    ny = len(thermo_y.options.species()) - 1
    print('ny = ', ny)

    w = block.buffer("hydro.eos.W")

    temp = Ts * torch.ones((nc3, nc2), dtype=w.dtype, device=w.device)
    pres = Ps * torch.ones((nc3, nc2), dtype=w.dtype, device=w.device)

    iH2O = thermo_y.options.species().index("H2O")
    print('iH2O = ', iH2O)

    xfrac = torch.zeros((nc3, nc2, ny + 1), dtype=w.dtype, device=w.device)
    xfrac[..., iH2O] = xH2O

    # dry air mole fraction
    xfrac[..., 0] = 1. - xH2O

    # start and end indices for the vertical direction
    ifirst = coord.ifirst()
    ilast = coord.ilast()

    # vertical grid distance of the first cell
    dz = coord.buffer("dx1f")[ifirst]

    # half a grid to cell center
    thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz);

    # adiabatic extrapolation
    for i in range(ifirst, ilast):
        conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))

        w[index.ipr, ..., i] = pres;
        w[index.idn, ..., i] = thermo_x.compute("V->D", (conc,))
        w[index.icy:,...,i] = thermo_x.compute("X->Y", (xfrac,))

        dz = coord.buffer("dx1f")[i]
        thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz);

    # initialize hydro state
    block.initialize(w)
    return w

if __name__ == '__main__':
    # input file
    infile = "earth.yaml"
    device = "cpu"

    # create meshblock
    op_block = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(op_block)
    block.to(torch.device(device))

    # create thermo module
    op_thermo = ThermoOptions.from_yaml(infile)
    thermo_x = ThermoX(op_thermo)
    thermo_x.to(torch.device(device))

    # create kinetics module
    op_kinet = KineticsOptions.from_yaml(infile)
    kinet = Kinetics(op_kinet)
    kinet.to(torch.device(device))

    # create output fields
    op_out = OutputOptions().file_basename("earth")
    out2 = NetcdfOutput(op_out.fid(2).variable("prim"))
    out3 = NetcdfOutput(op_out.fid(3).variable("uov"))
    out4 = NetcdfOutput(op_out.fid(4).variable("diag"))
    outs = [out2, out4]

    # set up initial condition
    w = setup_initial_condition(block, thermo_x)
    print("w = ", w[:,0,0,:])

    # integration
    current_time = 0.0
    count = 0
    start_time = time.time()
    interior = block.part((0, 0, 0))
    while not block.intg.stop(count, current_time):
        dt = block.max_time_step()
        u = block.buffer("hydro.eos.U")

        if count % 1 == 0:
            print(f"count = {count}, dt = {dt}, time = {current_time}")
            print("mass = ", u[interior][index.idn].sum())

            for out in outs:
                out.increment_file_number()
                out.write_output_file(block, current_time)
                out.combine_blocks()

        for stage in range(len(block.intg.stages)):
            block.forward(dt, stage)

        # evolve kinetics
        u[index.icy:] += evolve_kinetics(block, kinet, thermo_x)

        current_time += dt
        count += 1
