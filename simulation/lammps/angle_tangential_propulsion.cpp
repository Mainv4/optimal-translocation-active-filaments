/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include "angle_tangential_propulsion.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

AngleTangentialPropulsion::AngleTangentialPropulsion(LAMMPS *lmp) : Angle(lmp) {}

/* ---------------------------------------------------------------------- */

AngleTangentialPropulsion::~AngleTangentialPropulsion()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(Pe);
  }
}

/* ---------------------------------------------------------------------- */

void AngleTangentialPropulsion::compute(int eflag, int vflag)
{
  int i1,i2,i3,n,type;
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double eangle,f1[3],f3[3];
  double myFa;
  double rsq1,rsq2,r1,r2,c,s,a,a11,a12,a22;

  eangle = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    

    delx2 = x[i3][0] - x[i1][0];
    dely2 = x[i3][1] - x[i1][1];
    delz2 = x[i3][2] - x[i1][2];

    rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    r2 = sqrt(rsq2);

    

    myFa = Pe[type]/1.;

    

    if (newton_bond || i2 < nlocal) {
      f[i2][0] += myFa*delx2/r2;
      f[i2][1] += myFa*dely2/r2;
      f[i2][2] += myFa*delz2/r2;
    }

  }
}

/* ---------------------------------------------------------------------- */

void AngleTangentialPropulsion::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;

  memory->create(Pe,n+1,"angle:Pe");

  memory->create(setflag,n+1,"angle:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void AngleTangentialPropulsion::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Incorrect args for angle coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nangletypes,ilo,ihi,error);

  double Pe_one = utils::numeric(FLERR,arg[1],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    Pe[i] = Pe_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleTangentialPropulsion::equilibrium_angle(int i)
{
  return 180.;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void AngleTangentialPropulsion::write_restart(FILE *fp)
{
  fwrite(&Pe[1],sizeof(double),atom->nangletypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleTangentialPropulsion::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&Pe[1],sizeof(double),atom->nangletypes,fp);
  }
  MPI_Bcast(&Pe[1],atom->nangletypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleTangentialPropulsion::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp,"%d %g\n",i,Pe[i]);
}

/* ---------------------------------------------------------------------- */

double AngleTangentialPropulsion::single(int type, int i1, int i2, int i3)
{
  return 0.;
}
