/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2016 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Timo Heister, Clemson University, 2016
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>


#include <deal.II/lac/generic_linear_algebra.h>

/* #define FORCE_USE_OF_TRILINOS */

namespace LA
{
using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_qmrs.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>




#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>



#include <cmath>
#include <fstream>
#include <iostream>

namespace Step55
{
using namespace dealii;


namespace ChangeVectorTypes
{
void import(TrilinosWrappers::MPI::Vector &out,
            const dealii::LinearAlgebra::ReadWriteVector<double> &rwv,
            const VectorOperation::values                 operation)
{
  Assert(out.size() == rwv.size(),
         ExcMessage("Both vectors need to have the same size for import() to work!"));

  Assert(out.locally_owned_elements() == rwv.get_stored_elements(),
         ExcNotImplemented());

  if (operation == VectorOperation::insert)
  {
    for (const auto idx : out.locally_owned_elements())
      out[idx] = rwv[idx];
  }
  else if (operation == VectorOperation::add)
  {
    for (const auto idx : out.locally_owned_elements())
      out[idx] += rwv[idx];
  }
  else
    AssertThrow(false, ExcNotImplemented());

  out.compress(operation);
}


void copy(TrilinosWrappers::MPI::Vector &out,
          const dealii::LinearAlgebra::distributed::Vector<double> &in)
{
  dealii::LinearAlgebra::ReadWriteVector<double> rwv(out.locally_owned_elements());
  rwv.import(in, VectorOperation::insert);
  //This import function doesn't exist until after dealii 9.0
  //Implemented above
  import(out, rwv,VectorOperation::insert);
}

void copy(dealii::LinearAlgebra::distributed::Vector<double> &out,
          const TrilinosWrappers::MPI::Vector &in)
{
  dealii::LinearAlgebra::ReadWriteVector<double> rwv;
  rwv.reinit(in);
  out.import(rwv, VectorOperation::insert);
}

void copy(TrilinosWrappers::MPI::BlockVector &out,
          const dealii::LinearAlgebra::distributed::BlockVector<double> &in)
{
  const unsigned int n_blocks = in.n_blocks();
  for (unsigned int b=0; b<n_blocks; ++b)
    copy(out.block(b),in.block(b));
}

void copy(dealii::LinearAlgebra::distributed::BlockVector<double> &out,
          const TrilinosWrappers::MPI::BlockVector &in)
{
  const unsigned int n_blocks = in.n_blocks();
  for (unsigned int b=0; b<n_blocks; ++b)
    copy(out.block(b),in.block(b));
}
}
/**
 * This namespace contains all matrix-free operators used in the Stokes solver.
 */
namespace MatrixFreeStokesOperators
{
/**
   * Operator for the entire Stokes block.
   */
template <int dim, int degree_v, typename number>
class StokesOperator
    : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >
{
public:

  /**
       * Constructor.
       */
  StokesOperator ();

  /**
       * Reset the viscosity table.
       */
  void clear ();

  /**
       * Fills in the viscosity table and set the value for the pressure scaling constant.
       */
  void fill_viscosities_and_pressure_scaling(const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                                             const double pressure_scaling,
                                             const Triangulation<dim> &tria,
                                             const DoFHandler<dim> &dof_handler_for_projection);

  /**
       * Returns the viscosity table.
       */
  const Table<2, VectorizedArray<number> > &
  get_viscosity_x_2_table();

  /**
       * Computes the diagonal of the matrix. Since matrix-free operators have not access
       * to matrix elements, we must apply the matrix-free operator to the unit vectors to
       * recover the diagonal.
       */
  virtual void compute_diagonal ();

private:

  /**
       * Performs the application of the matrix-free operator. This function is called by
       * vmult() functions MatrixFreeOperators::Base.
       */
  virtual void apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                          const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const;

  /**
       * Defines the application of the cell matrix.
       */
  void local_apply (const dealii::MatrixFree<dim, number> &data,
                    dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                    const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range) const;

  /**
       * Table which stores the viscosity on each quadrature point.
       */
  Table<2, VectorizedArray<number> > viscosity_x_2;

  /**
       * Pressure scaling constant.
       */
  double pressure_scaling;
};

/**
   * Operator for the pressure mass matrix used in the block preconditioner
   */
template <int dim, int degree_p, typename number>
class MassMatrixOperator
    : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>
{
public:

  /**
       * Constructor
       */
  MassMatrixOperator ();

  /**
       * Reset the viscosity table.
       */
  void clear ();

  /**
       * Fills in the viscosity table and set the value for the pressure scaling constant.
       */
  void fill_viscosities_and_pressure_scaling (const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                                              const double pressure_scaling,
                                              const Triangulation<dim> &tria,
                                              const DoFHandler<dim> &dof_handler_for_projection);


  /**
       * Computes the diagonal of the matrix. Since matrix-free operators have not access
       * to matrix elements, we must apply the matrix-free operator to the unit vectors to
       * recover the diagonal.
       */
  virtual void compute_diagonal ();

private:

  /**
       * Performs the application of the matrix-free operator. This function is called by
       * vmult() functions MatrixFreeOperators::Base.
       */
  virtual void apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                          const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

  /**
       * Defines the application of the cell matrix.
       */
  void local_apply (const dealii::MatrixFree<dim, number> &data,
                    dealii::LinearAlgebra::distributed::Vector<number> &dst,
                    const dealii::LinearAlgebra::distributed::Vector<number> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range) const;


  /**
       * Computes the diagonal contribution from a cell matrix.
       */
  void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                               dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                               const unsigned int                               &dummy,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const;

  /**
       * Table which stores the viscosity on each quadrature point.
       */
  Table<2, VectorizedArray<number> > one_over_viscosity;

  /**
       * Pressure scaling constant.
       */
  double pressure_scaling;
};

/**
   * Operator for the A block of the Stokes matrix. The same class is used for both
   * active and level mesh operators.
   */
template <int dim, int degree_v, typename number>
class ABlockOperator
    : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>
{
public:

  /**
       * Constructor
       */
  ABlockOperator ();

  /**
       * Reset the viscosity table.
       */
  void clear ();

  /**
       * Fills in the viscosity table.
       */
  void fill_viscosities(const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                        const Triangulation<dim> &tria,
                        const DoFHandler<dim> &dof_handler_for_projection,
                        const bool for_mg);

  /**
       * Computes the diagonal of the matrix. Since matrix-free operators have not access
       * to matrix elements, we must apply the matrix-free operator to the unit vectors to
       * recover the diagonal.
       */
  virtual void compute_diagonal ();

  /**
              *
              */
  void set_diagonal (const dealii::LinearAlgebra::distributed::Vector<number> &diag);

private:

  /**
       * Performs the application of the matrix-free operator. This function is called by
       * vmult() functions MatrixFreeOperators::Base.
       */
  virtual void apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                          const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

  /**
       * Defines the application of the cell matrix.
       */
  void local_apply (const dealii::MatrixFree<dim, number> &data,
                    dealii::LinearAlgebra::distributed::Vector<number> &dst,
                    const dealii::LinearAlgebra::distributed::Vector<number> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range) const;

  /**
       * Computes the diagonal contribution from a cell matrix.
       */
  void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                               dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                               const unsigned int                               &dummy,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const;

  /**
       * Table which stores the viscosity on each quadrature point.
       */
  Table<2, VectorizedArray<number> > viscosity_x_2;

};

/**
 * Implementation of the matrix-free operators.
 *
 * Stokes operator
 */
template <int dim, int degree_v, typename number>
StokesOperator<dim,degree_v,number>::StokesOperator ()
  :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >()
{}

template <int dim, int degree_v, typename number>
void
StokesOperator<dim,degree_v,number>::clear ()
{
  viscosity_x_2.reinit(0, 0);
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::BlockVector<number> >::clear();
}

template <int dim, int degree_v, typename number>
void
StokesOperator<dim,degree_v,number>::
fill_viscosities_and_pressure_scaling (const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                                       const double pressure_scaling,
                                       const Triangulation<dim> &tria,
                                       const DoFHandler<dim> &dof_handler_for_projection)
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (*this->data, 0);
  const unsigned int n_cells = this->data->n_macro_cells();
  viscosity_x_2.reinit(n_cells, velocity.n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dof_handler_for_projection.get_fe().dofs_per_cell);
  for (unsigned int cell=0; cell<n_cells; ++cell)
    for (unsigned int i=0; i<this->get_matrix_free()->n_components_filled(cell); ++i)
    {
      typename DoFHandler<dim>::active_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
      typename DoFHandler<dim>::active_cell_iterator DG_cell(&tria,
                                                             FEQ_cell->level(),
                                                             FEQ_cell->index(),
                                                             &dof_handler_for_projection);
      DG_cell->get_active_or_mg_dof_indices(local_dof_indices);

      //TODO: projection with higher degree
      Assert(local_dof_indices.size() == 1, ExcNotImplemented());
      for (unsigned int q=0; q<velocity.n_q_points; ++q)
        viscosity_x_2(cell,q)[i] = 2.0*viscosity_values(local_dof_indices[0]);
    }
  this->pressure_scaling = pressure_scaling;
}

template <int dim, int degree_v, typename number>
const Table<2, VectorizedArray<number> > &
StokesOperator<dim,degree_v,number>::get_viscosity_x_2_table()
{
  return viscosity_x_2;
}

template <int dim, int degree_v, typename number>
void
StokesOperator<dim,degree_v,number>
::compute_diagonal ()
{
  // There is currently no need in the code for the diagonal of the entire stokes
  // block. If needed, one could easily construct based on the diagonal of the A
  // block and append zeros to the end for the number of pressure DoFs.
  Assert(false, ExcNotImplemented());
}

template <int dim, int degree_v, typename number>
void
StokesOperator<dim,degree_v,number>
::local_apply (const dealii::MatrixFree<dim, number>                 &data,
               dealii::LinearAlgebra::distributed::BlockVector<number>       &dst,
               const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
               const std::pair<unsigned int, unsigned int>           &cell_range) const
{
  typedef VectorizedArray<number> vector_t;
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data, 0);
  FEEvaluation<dim,degree_v-1,  degree_v+1,1,  number> pressure (data, 1);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);
    velocity.read_dof_values (src.block(0));
    velocity.evaluate (false,true,false);
    pressure.reinit (cell);
    pressure.read_dof_values (src.block(1));
    pressure.evaluate (true,false,false);

    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      SymmetricTensor<2,dim,vector_t> sym_grad_u =
          velocity.get_symmetric_gradient (q);
      vector_t pres = pressure.get_value(q);
      vector_t div = -trace(sym_grad_u);
      pressure.submit_value   (pressure_scaling*div, q);

      sym_grad_u *= viscosity_x_2(cell,q);

      for (unsigned int d=0; d<dim; ++d)
        sym_grad_u[d][d] -= pressure_scaling*pres;

      velocity.submit_symmetric_gradient(sym_grad_u, q);
    }

    velocity.integrate (false,true);
    velocity.distribute_local_to_global (dst.block(0));
    pressure.integrate (true,false);
    pressure.distribute_local_to_global (dst.block(1));
  }
}

template <int dim, int degree_v, typename number>
void
StokesOperator<dim,degree_v,number>
::apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
             const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const
{
  MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >::
      data->cell_loop(&StokesOperator::local_apply, this, dst, src);
}

/**
 * Mass matrix operator
 */
template <int dim, int degree_p, typename number>
MassMatrixOperator<dim,degree_p,number>::MassMatrixOperator ()
  :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number> >()
{}

template <int dim, int degree_p, typename number>
void
MassMatrixOperator<dim,degree_p,number>::clear ()
{
  one_over_viscosity.reinit(0, 0);
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::clear();
}

template <int dim, int degree_p, typename number>
void
MassMatrixOperator<dim,degree_p,number>::
fill_viscosities_and_pressure_scaling (const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                                       const double pressure_scaling,
                                       const Triangulation<dim> &tria,
                                       const DoFHandler<dim> &dof_handler_for_projection)
{
  FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (*this->data, 0);
  const unsigned int n_cells = this->data->n_macro_cells();
  one_over_viscosity.reinit(n_cells, pressure.n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dof_handler_for_projection.get_fe().dofs_per_cell);
  for (unsigned int cell=0; cell<n_cells; ++cell)
    for (unsigned int i=0; i<this->get_matrix_free()->n_components_filled(cell); ++i)
    {
      typename DoFHandler<dim>::active_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
      typename DoFHandler<dim>::active_cell_iterator DG_cell(&tria,
                                                             FEQ_cell->level(),
                                                             FEQ_cell->index(),
                                                             &dof_handler_for_projection);
      DG_cell->get_active_or_mg_dof_indices(local_dof_indices);

      //TODO: projection with higher degree
      Assert(local_dof_indices.size() == 1, ExcNotImplemented());
      for (unsigned int q=0; q<pressure.n_q_points; ++q)
        one_over_viscosity(cell,q)[i] = 1.0/viscosity_values(local_dof_indices[0]);
    }
  this->pressure_scaling = pressure_scaling;
}

template <int dim, int degree_p, typename number>
void
MassMatrixOperator<dim,degree_p,number>
::local_apply (const dealii::MatrixFree<dim, number>                 &data,
               dealii::LinearAlgebra::distributed::Vector<number>       &dst,
               const dealii::LinearAlgebra::distributed::Vector<number> &src,
               const std::pair<unsigned int, unsigned int>           &cell_range) const
{
  FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (data);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    AssertDimension(one_over_viscosity.size(0), data.n_macro_cells());
    AssertDimension(one_over_viscosity.size(1), pressure.n_q_points);

    pressure.reinit (cell);
    pressure.read_dof_values(src);
    pressure.evaluate (true, false);
    for (unsigned int q=0; q<pressure.n_q_points; ++q)
      pressure.submit_value(one_over_viscosity(cell,q)*pressure_scaling*pressure_scaling*
                            pressure.get_value(q),q);
    pressure.integrate (true, false);
    pressure.distribute_local_to_global (dst);
  }
}

template <int dim, int degree_p, typename number>
void
MassMatrixOperator<dim,degree_p,number>
::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
             const dealii::LinearAlgebra::distributed::Vector<number> &src) const
{
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::
      data->cell_loop(&MassMatrixOperator::local_apply, this, dst, src);
}

template <int dim, int degree_p, typename number>
void
MassMatrixOperator<dim,degree_p,number>
::compute_diagonal ()
{
  this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());
  this->diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());

  dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
  dealii::LinearAlgebra::distributed::Vector<number> &diagonal =
      this->diagonal_entries->get_vector();

  unsigned int dummy = 0;
  this->data->initialize_dof_vector(inverse_diagonal);
  this->data->initialize_dof_vector(diagonal);

  this->data->cell_loop (&MassMatrixOperator::local_compute_diagonal, this,
                         diagonal, dummy);

  this->set_constrained_entries_to_one(diagonal);
  inverse_diagonal = diagonal;
  const unsigned int local_size = inverse_diagonal.local_size();
  for (unsigned int i=0; i<local_size; ++i)
  {
    Assert(inverse_diagonal.local_element(i) > 0.,
           ExcMessage("No diagonal entry in a positive definite operator "
                      "should be zero"));
    inverse_diagonal.local_element(i)
        =1./inverse_diagonal.local_element(i);
  }
}

template <int dim, int degree_p, typename number>
void
MassMatrixOperator<dim,degree_p,number>
::local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                          dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                          const unsigned int &,
                          const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (data, 0);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    pressure.reinit (cell);
    AlignedVector<VectorizedArray<number> > diagonal(pressure.dofs_per_cell);
    for (unsigned int i=0; i<pressure.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<pressure.dofs_per_cell; ++j)
        pressure.begin_dof_values()[j] = VectorizedArray<number>();
      pressure.begin_dof_values()[i] = make_vectorized_array<number> (1.);

      pressure.evaluate (true,false,false);
      for (unsigned int q=0; q<pressure.n_q_points; ++q)
        pressure.submit_value(one_over_viscosity(cell,q)*pressure_scaling*pressure_scaling*
                              pressure.get_value(q),q);
      pressure.integrate (true,false);

      diagonal[i] = pressure.begin_dof_values()[i];
    }

    for (unsigned int i=0; i<pressure.dofs_per_cell; ++i)
      pressure.begin_dof_values()[i] = diagonal[i];
    pressure.distribute_local_to_global (dst);
  }
}

/**
 * Velocity block operator
 */
template <int dim, int degree_v, typename number>
ABlockOperator<dim,degree_v,number>::ABlockOperator ()
  :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number> >()
{}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>::clear ()
{
  viscosity_x_2.reinit(0, 0);
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::clear();
}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>::
fill_viscosities (const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                  const Triangulation<dim> &tria,
                  const DoFHandler<dim> &dof_handler_for_projection,
                  const bool for_mg)
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (*this->data, 0);
  const unsigned int n_cells = this->data->n_macro_cells();
  viscosity_x_2.reinit(n_cells, velocity.n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dof_handler_for_projection.get_fe().dofs_per_cell);
  for (unsigned int cell=0; cell<n_cells; ++cell)
    for (unsigned int i=0; i<this->get_matrix_free()->n_components_filled(cell); ++i)
    {

      if (for_mg)
      {
        typename DoFHandler<dim>::level_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
        typename DoFHandler<dim>::level_cell_iterator DG_cell(&tria,
                                                              FEQ_cell->level(),
                                                              FEQ_cell->index(),
                                                              &dof_handler_for_projection);
        DG_cell->get_active_or_mg_dof_indices(local_dof_indices);
      }
      else
      {
        typename DoFHandler<dim>::active_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
        typename DoFHandler<dim>::active_cell_iterator DG_cell(&tria,
                                                               FEQ_cell->level(),
                                                               FEQ_cell->index(),
                                                               &dof_handler_for_projection);
        DG_cell->get_active_or_mg_dof_indices(local_dof_indices);
      }

      //TODO: projection with higher degree
      Assert(local_dof_indices.size() == 1, ExcNotImplemented());
      for (unsigned int q=0; q<velocity.n_q_points; ++q)
        viscosity_x_2(cell,q)[i] = 2.0*viscosity_values(local_dof_indices[0]);
    }
}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::local_apply (const dealii::MatrixFree<dim, number>                 &data,
               dealii::LinearAlgebra::distributed::Vector<number>       &dst,
               const dealii::LinearAlgebra::distributed::Vector<number> &src,
               const std::pair<unsigned int, unsigned int>           &cell_range) const
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    AssertDimension(viscosity_x_2.size(0), data.n_macro_cells());
    AssertDimension(viscosity_x_2.size(1), velocity.n_q_points);

    velocity.reinit (cell);
    velocity.read_dof_values(src);
    velocity.evaluate (false, true, false);
    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      velocity.submit_symmetric_gradient
          (viscosity_x_2(cell,q)*velocity.get_symmetric_gradient(q),q);
    }
    velocity.integrate (false, true);
    velocity.distribute_local_to_global (dst);
  }
}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
             const dealii::LinearAlgebra::distributed::Vector<number> &src) const
{
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::
      data->cell_loop(&ABlockOperator::local_apply, this, dst, src);
}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::compute_diagonal ()
{
  this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());
  dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);
  unsigned int dummy = 0;
  this->data->cell_loop (&ABlockOperator::local_compute_diagonal, this,
                         inverse_diagonal, dummy);

  this->set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
  {
    Assert(inverse_diagonal.local_element(i) > 0.,
           ExcMessage("No diagonal entry in a positive definite operator "
                      "should be zero"));
    inverse_diagonal.local_element(i) =
        1./inverse_diagonal.local_element(i);
  }
}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                          dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                          const unsigned int &,
                          const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data, 0);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);
    AlignedVector<VectorizedArray<number> > diagonal(velocity.dofs_per_cell);
    for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<velocity.dofs_per_cell; ++j)
        velocity.begin_dof_values()[j] = VectorizedArray<number>();
      velocity.begin_dof_values()[i] = make_vectorized_array<number> (1.);

      velocity.evaluate (false,true,false);
      for (unsigned int q=0; q<velocity.n_q_points; ++q)
      {
        velocity.submit_symmetric_gradient
            (viscosity_x_2(cell,q)*velocity.get_symmetric_gradient(q),q);
      }
      velocity.integrate (false,true);

      diagonal[i] = velocity.begin_dof_values()[i];
    }

    for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
      velocity.begin_dof_values()[i] = diagonal[i];
    velocity.distribute_local_to_global (dst);
  }
}


template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::set_diagonal (const dealii::LinearAlgebra::distributed::Vector<number> &diag)
{
  this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());
  dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);

  inverse_diagonal = diag;

  this->set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
  {
    Assert(inverse_diagonal.local_element(i) > 0.,
           ExcMessage("No diagonal entry in a positive definite operator "
                      "should be zero"));
    inverse_diagonal.local_element(i) =
        1./inverse_diagonal.local_element(i);
  }
}
}

namespace LinearSolvers
{
template <class Matrix, class Preconditioner>
class InverseMatrix : public Subscriptor
{
public:
  InverseMatrix(const Matrix &m, const Preconditioner &preconditioner);

  void vmult(dealii::LinearAlgebra::distributed::Vector<double> &dst,
             const dealii::LinearAlgebra::distributed::Vector<double> &src) const;

private:
  const SmartPointer<const Matrix> matrix;
  const Preconditioner &           preconditioner;
};


template <class Matrix, class Preconditioner>
InverseMatrix<Matrix, Preconditioner>::InverseMatrix(
    const Matrix &        m,
    const Preconditioner &preconditioner)
  : matrix(&m)
  , preconditioner(preconditioner)
{}



template <class Matrix, class Preconditioner>
void
InverseMatrix<Matrix, Preconditioner>::vmult(dealii::LinearAlgebra::distributed::Vector<double> &      dst,
                                             const dealii::LinearAlgebra::distributed::Vector<double> &src) const
{
  SolverControl solver_control(src.size(), 1e-8 * src.l2_norm());
  SolverCG<dealii::LinearAlgebra::distributed::Vector<double>> cg(solver_control);
  dst = 0;

  try
  {
    cg.solve(*matrix, dst, src, preconditioner);
  }
  catch (std::exception &e)
  {
    Assert(false, ExcMessage(e.what()));
  }
}


template <class PreconditionerA, class PreconditionerS>
class BlockDiagonalPreconditioner : public Subscriptor
{
public:
  BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                              const PreconditionerS &preconditioner_S);

  void vmult(dealii::LinearAlgebra::distributed::BlockVector<double> &      dst,
             const dealii::LinearAlgebra::distributed::BlockVector<double> &src) const;

private:
  const PreconditionerA &preconditioner_A;
  const PreconditionerS &preconditioner_S;
};

template <class PreconditionerA, class PreconditionerS>
BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                            const PreconditionerS &preconditioner_S)
  : preconditioner_A(preconditioner_A)
  , preconditioner_S(preconditioner_S)
{}


template <class PreconditionerA, class PreconditionerS>
void BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::vmult(
    dealii::LinearAlgebra::distributed::BlockVector<double> &      dst,
    const dealii::LinearAlgebra::distributed::BlockVector<double> &src) const
{
  preconditioner_A.vmult(dst.block(0), src.block(0));
  preconditioner_S.vmult(dst.block(1), src.block(1));
}

} // namespace LinearSolvers



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>(dim + 1)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &  value) const override;
};


template <int dim>
void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &  values) const
{
  const double R_x = p[0];
  const double R_y = p[1];
  const double R_z = 1.0;

  const double pi  = numbers::PI;
  const double pi2 = pi * pi;
  values[0] =
      -1.0L / 2.0L * (-2 * sqrt(25.0 + 4 * pi2) + 10.0) *
      exp(R_x * (-2 * sqrt(25.0 + 4 * pi2) + 10.0)) -
      0.4 * pi2 * exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi) +
      0.1 * pow(-sqrt(25.0 + 4 * pi2) + 5.0, 2) *
      exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi);
  values[1] = 0.2 * pi * (-sqrt(25.0 + 4 * pi2) + 5.0) *
      exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) -
      0.05 * pow(-sqrt(25.0 + 4 * pi2) + 5.0, 3) *
      exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) /
      pi;
  values[2] = R_z;
  values[3] = 0.;
}


template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution()
    : Function<dim>(dim + 1)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &  value) const override;
};

template <int dim>
void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &  values) const
{
  const double R_x = p[0];
  const double R_y = p[1];
  const double R_z = 1.0;

  const double pi  = numbers::PI;
  const double pi2 = pi * pi;
  values[0] =
      -exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi) + 1;
  values[1] = (1.0L / 2.0L) * (-sqrt(25.0 + 4 * pi2) + 5.0) *
      exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) /
      pi;
  values[2] = R_z;
  values[3] =
      -1.0L / 2.0L * exp(R_x * (-2 * sqrt(25.0 + 4 * pi2) + 10.0)) -
      2.0 *
      (-6538034.74494422 +
       0.0134758939981709 * exp(4 * sqrt(25.0 + 4 * pi2))) /
      (-80.0 * exp(3 * sqrt(25.0 + 4 * pi2)) +
       16.0 * sqrt(25.0 + 4 * pi2) * exp(3 * sqrt(25.0 + 4 * pi2))) -
      1634508.68623606 * exp(-3.0 * sqrt(25.0 + 4 * pi2)) /
      (-10.0 + 2.0 * sqrt(25.0 + 4 * pi2)) +
      (-0.00673794699908547 * exp(sqrt(25.0 + 4 * pi2)) +
       3269017.37247211 * exp(-3 * sqrt(25.0 + 4 * pi2))) /
      (-8 * sqrt(25.0 + 4 * pi2) + 40.0) +
      0.00336897349954273 * exp(1.0 * sqrt(25.0 + 4 * pi2)) /
      (-10.0 + 2.0 * sqrt(25.0 + 4 * pi2));
}

template <int dim>
class ExactSolution_v : public Function<dim>
{
public:
  ExactSolution_v()
    : Function<dim>(dim)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &  value) const override;
};

template <int dim>
void ExactSolution_v<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
{
  const double R_x = p[0];
  const double R_y = p[1];
  const double R_z = 1.0;

  const double pi  = numbers::PI;
  const double pi2 = pi * pi;
  values[0] =
      -exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi) + 1;
  values[1] = (1.0L / 2.0L) * (-sqrt(25.0 + 4 * pi2) + 5.0) *
      exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) /
      pi;
  values[2] = R_z;
}


template <int dim>
class StokesProblem
{
public:
  StokesProblem(unsigned int velocity_degree);

  void run(unsigned int n_cycles);

private:
  void make_grid();
  void setup_system();
  void assemble_system();

  void evaluate_viscosity();
  void correct_stokes_rhs();

  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  unsigned int velocity_degree;
  double       viscosity;
  MPI_Comm     mpi_communicator;

  FESystem<dim>                             fe;
  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim>                           dof_handler;

  std::vector<IndexSet> owned_partitioning;
  std::vector<IndexSet> relevant_partitioning;

  AffineConstraints<double> constraints;

  LA::MPI::BlockVector       locally_relevant_solution;
  LA::MPI::BlockVector       system_rhs;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;




  DoFHandler<dim> dof_handler_v;
  DoFHandler<dim> dof_handler_p;
  DoFHandler<dim> dof_handler_projection;

  FESystem<dim> stokes_fe;
  FESystem<dim> fe_v;
  FESystem<dim> fe_p;
  FESystem<dim> fe_projection;

  // TODO: velocity degree not only 2, Choosing quadrature degree?
  typedef MatrixFreeStokesOperators::StokesOperator<dim,2,double> StokesMatrixType;
  typedef MatrixFreeStokesOperators::MassMatrixOperator<dim,1,double> MassMatrixType;
  typedef MatrixFreeStokesOperators::ABlockOperator<dim,2,double> ABlockMatrixType;

  StokesMatrixType stokes_matrix;
  ABlockMatrixType velocity_matrix;
  MassMatrixType mass_matrix;

  AffineConstraints<double> constraints_v;
  AffineConstraints<double> constraints_p;
  AffineConstraints<double> constraints_projection;

  MGLevelObject<ABlockMatrixType> mg_matrices;
  MGConstrainedDoFs              mg_constrained_dofs;
  MGConstrainedDoFs mg_constrained_dofs_projection;

  dealii::LinearAlgebra::distributed::Vector<double> active_coef_dof_vec;
  MGLevelObject<dealii::LinearAlgebra::distributed::Vector<double> > level_coef_dof_vec;


  MGTransferMatrixFree<dim,double> mg_transfer;
};



template <int dim>
StokesProblem<dim>::StokesProblem(unsigned int velocity_degree)
  : velocity_degree(velocity_degree)
  , viscosity(0.1)
  , mpi_communicator(MPI_COMM_WORLD)
  , fe(FE_Q<dim>(velocity_degree), dim, FE_Q<dim>(velocity_degree - 1), 1)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening |
                    Triangulation<dim>::limit_level_difference_at_vertices),
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
  , dof_handler(triangulation)
  , pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times),




    dof_handler_v(triangulation),
    dof_handler_p(triangulation),
    dof_handler_projection(triangulation),

    stokes_fe (FE_Q<dim>(velocity_degree),dim,
               FE_Q<dim>(velocity_degree-1),1),
    fe_v (FE_Q<dim>(velocity_degree), dim),
    fe_p (FE_Q<dim>(velocity_degree-1),1),
    fe_projection(FE_DGQ<dim>(0),1)
{}


template <int dim>
void StokesProblem<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -0.5, 1.5);
  triangulation.refine_global(3);
}

template <int dim>
void StokesProblem<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "1.setup");

  {
    TimerOutput::Scope t(computing_timer, "setup:distribute_dofs");
    dof_handler.distribute_dofs(fe);
  }

  std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
  stokes_sub_blocks[dim] = 1;

  {
    TimerOutput::Scope t(computing_timer, "setup:renumber");
    DoFRenumbering::component_wise(dof_handler, stokes_sub_blocks);
  }

  std::vector<types::global_dof_index> dofs_per_block(2);
  DoFTools::count_dofs_per_block(dof_handler,
                                 dofs_per_block,
                                 stokes_sub_blocks);

  const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1];

  pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
        << n_u << '+' << n_p << ')' << std::endl;

  owned_partitioning.resize(2);
  owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_u);
  owned_partitioning[1] =
      dof_handler.locally_owned_dofs().get_view(n_u, n_u + n_p);

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  relevant_partitioning.resize(2);
  relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
  relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

  {
    TimerOutput::Scope t(computing_timer, "setup:constraints");
    constraints.reinit(locally_relevant_dofs);

    FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             ExactSolution<dim>(),
                                             constraints,
                                             fe.component_mask(velocities));
    constraints.close();
  }

  locally_relevant_solution.reinit(owned_partitioning,
                                   relevant_partitioning,
                                   mpi_communicator);
  system_rhs.reinit(owned_partitioning, mpi_communicator);





  // Velocity DoFHandler
  {
    TimerOutput::Scope t(computing_timer, "setup:velocity_dofh");
    dof_handler_v.clear();
    dof_handler_v.distribute_dofs(fe_v);

    DoFRenumbering::hierarchical(dof_handler_v);

    constraints_v.clear();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs (dof_handler_v,
                                             locally_relevant_dofs);
    constraints_v.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler_v, constraints_v);
    VectorTools::interpolate_boundary_values(dof_handler_v,
                                             0,
                                             ExactSolution_v<dim>(),
                                             constraints_v);
    constraints_v.close ();
  }

  // Pressure DoFHandler
  {
    TimerOutput::Scope t(computing_timer, "setup:pressure_dofh");
    dof_handler_p.clear();
    dof_handler_p.distribute_dofs(fe_p);

    DoFRenumbering::hierarchical(dof_handler_p);

    constraints_p.clear();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs (dof_handler_p,
                                             locally_relevant_dofs);
    constraints_p.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler_p, constraints_p);
    constraints_p.close();
  }

  // Coefficient transfer objects
  {
    TimerOutput::Scope t(computing_timer, "setup:transfer_dofh");
    dof_handler_projection.clear();
    dof_handler_projection.distribute_dofs(fe_projection);

    DoFRenumbering::hierarchical(dof_handler_projection);

    active_coef_dof_vec.reinit(dof_handler_projection.locally_owned_dofs(), mpi_communicator);
  }

  // Multigrid DoF setup

  {
    TimerOutput::Scope t(computing_timer, "setup:distribute_mg_dofs");
    dof_handler_v.distribute_mg_dofs();
  }

  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler_v);

  std::set<types::boundary_id> dirichlet_boundary;
  dirichlet_boundary.insert(0);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_v, dirichlet_boundary);

  {
    TimerOutput::Scope t(computing_timer, "setup:transfer:distribute_mg_dofs");
    dof_handler_projection.distribute_mg_dofs();
  }

  // Setup the matrix-free operators
  // Stokes matrix
  {
    TimerOutput::Scope t(computing_timer, "setup:matrix-free");
    typename MatrixFree<dim,double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim,double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_values | update_gradients |
                                            update_JxW_values | update_quadrature_points);

    std::vector<const DoFHandler<dim>*> stokes_dofs;
    stokes_dofs.push_back(&dof_handler_v);
    stokes_dofs.push_back(&dof_handler_p);
    std::vector<const AffineConstraints<double> *> stokes_constraints;
    stokes_constraints.push_back(&constraints_v);
    stokes_constraints.push_back(&constraints_p);

    std::shared_ptr<MatrixFree<dim,double> >
        stokes_mf_storage(new MatrixFree<dim,double>());
    stokes_mf_storage->reinit(stokes_dofs, stokes_constraints,
                              QGauss<1>(velocity_degree+1), additional_data);
    stokes_matrix.clear();
    stokes_matrix.initialize(stokes_mf_storage);

  }

  // ABlock matrix
  {
    typename MatrixFree<dim,double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim,double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_values | update_gradients |
                                            update_JxW_values | update_quadrature_points);
    std::shared_ptr<MatrixFree<dim,double> >
        ablock_mf_storage(new MatrixFree<dim,double>());
    ablock_mf_storage->reinit(dof_handler_v, constraints_v,
                              QGauss<1>(velocity_degree+1), additional_data);

    velocity_matrix.clear();
    velocity_matrix.initialize(ablock_mf_storage);
  }

  // Mass matrix
  {
    typename MatrixFree<dim,double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim,double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_values | update_JxW_values |
                                            update_quadrature_points);
    std::shared_ptr<MatrixFree<dim,double> >
        mass_mf_storage(new MatrixFree<dim,double>());
    mass_mf_storage->reinit(dof_handler_p, constraints_p,
                            QGauss<1>(velocity_degree+1), additional_data);

    mass_matrix.clear();
    mass_matrix.initialize(mass_mf_storage);
  }

  // GMG matrices
  {
    TimerOutput::Scope t(computing_timer, "setup:gmg-matrices");
    const unsigned int n_levels = triangulation.n_global_levels();
    mg_matrices.clear_elements();
    mg_matrices.resize(0, n_levels-1);

    for (unsigned int level=0; level<n_levels; ++level)
    {
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler_v, level, relevant_dofs);

      AffineConstraints<double> level_constraints;
      level_constraints.reinit(relevant_dofs);
      level_constraints.add_lines (mg_constrained_dofs.get_boundary_indices(level));
      level_constraints.close();

      {
        typename MatrixFree<dim,double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
            MatrixFree<dim,double>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                update_quadrature_points);
        additional_data.level_mg_handler = level;
        std::shared_ptr<MatrixFree<dim,double> >
            mg_mf_storage_level(new MatrixFree<dim,double>());
        mg_mf_storage_level->reinit(dof_handler_v, level_constraints,
                                    QGauss<1>(velocity_degree+1),
                                    additional_data);

        mg_matrices[level].clear();
        mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);

      }
    }
  }

  // Build MG transfer

  {
    TimerOutput::Scope t(computing_timer, "setup:mg_transfer");
    mg_transfer.clear();
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.build(dof_handler_v);
  }

}



template <int dim>
void StokesProblem<dim>::assemble_system()
{
  TimerOutput::Scope t(computing_timer, "2.assembly");

  system_rhs            = 0;

  const QGauss<dim> quadrature_formula(velocity_degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double>     cell_rhs(dofs_per_cell);

  const RightHandSide<dim>    right_hand_side;
  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      cell_rhs     = 0;

      fe_values.reinit(cell);
      right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                        rhs_values);
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int component_i =
              fe.system_to_component_index(i).first;
          cell_rhs(i) += fe_values.shape_value(i, q) *
              rhs_values[q](component_i) * fe_values.JxW(q);
        }
      }


      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_rhs,
                                             local_dof_indices,
                                             system_rhs);
    }

  system_rhs.compress(VectorOperation::add);
}



template <int dim>
void StokesProblem<dim>::evaluate_viscosity ()
{
  TimerOutput::Scope t(computing_timer, "3.evaluate_viscosity");

  {
    const QGauss<dim> quadrature_formula (velocity_degree+1);

    std::vector<types::global_dof_index> local_dof_indices(fe_projection.dofs_per_cell);
    active_coef_dof_vec = 0.;

    // compute the integral quantities by quadrature
    for (const auto &cell: dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        typename DoFHandler<dim>::active_cell_iterator dg_cell(&triangulation,
                                                               cell->level(),
                                                               cell->index(),
                                                               &dof_handler_projection);
        dg_cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < fe_projection.dofs_per_cell; ++i)
          active_coef_dof_vec[local_dof_indices[i]] = viscosity;
      }
    active_coef_dof_vec.compress(VectorOperation::insert);
  }

  stokes_matrix.fill_viscosities_and_pressure_scaling(active_coef_dof_vec,
                                                      1.0,
                                                      triangulation,
                                                      dof_handler_projection);

  velocity_matrix.fill_viscosities(active_coef_dof_vec,
                                   triangulation,
                                   dof_handler_projection,
                                   false);

  mass_matrix.fill_viscosities_and_pressure_scaling(active_coef_dof_vec,
                                                    1.0,
                                                    triangulation,
                                                    dof_handler_projection);
  mass_matrix.compute_diagonal();


  // Project to MG
  const unsigned int n_levels = triangulation.n_global_levels();
  level_coef_dof_vec = 0.;
  level_coef_dof_vec.resize(0,n_levels-1);

  MGTransferMatrixFree<dim,double> transfer(mg_constrained_dofs);
  transfer.build(dof_handler_projection);
  transfer.interpolate_to_mg(dof_handler_projection,
                             level_coef_dof_vec,
                             active_coef_dof_vec);

  for (unsigned int level=0; level<n_levels; ++level)
  {
    mg_matrices[level].fill_viscosities(level_coef_dof_vec[level],
                                        triangulation,
                                        dof_handler_projection,
                                        true);
    mg_matrices[level].compute_diagonal();
  }
}


template <int dim>
void StokesProblem<dim>::correct_stokes_rhs()
{
  TimerOutput::Scope t(computing_timer, "4.correct_stokes_rhs");
  dealii::LinearAlgebra::distributed::BlockVector<double> rhs_correction(2);
  dealii::LinearAlgebra::distributed::BlockVector<double> u0(2);

  stokes_matrix.initialize_dof_vector(rhs_correction);
  stokes_matrix.initialize_dof_vector(u0);

  rhs_correction.collect_sizes();
  u0.collect_sizes();

  u0 = 0;
  rhs_correction = 0;
  constraints.distribute(u0);
  u0.update_ghost_values();

  const Table<2, VectorizedArray<double>> viscosity_x_2_table = stokes_matrix.get_viscosity_x_2_table();
  FEEvaluation<dim,2,3,dim,double>
      velocity (*stokes_matrix.get_matrix_free(), 0);
  FEEvaluation<dim,1,3,1,double>
      pressure (*stokes_matrix.get_matrix_free(), 1);

  for (unsigned int cell=0; cell<stokes_matrix.get_matrix_free()->n_macro_cells(); ++cell)
  {
    velocity.reinit (cell);
    velocity.read_dof_values_plain (u0.block(0));
    velocity.evaluate (false,true,false);
    pressure.reinit (cell);
    pressure.read_dof_values_plain (u0.block(1));
    pressure.evaluate (true,false,false);

    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      SymmetricTensor<2,dim,VectorizedArray<double>> sym_grad_u =
          velocity.get_symmetric_gradient (q);
      VectorizedArray<double> pres = pressure.get_value(q);
      VectorizedArray<double> div = -trace(sym_grad_u);
      pressure.submit_value   (-1.0*div, q);

      sym_grad_u *= viscosity_x_2_table(cell,q);

      for (unsigned int d=0; d<dim; ++d)
        sym_grad_u[d][d] -= pres;

      velocity.submit_symmetric_gradient(-1.0*sym_grad_u, q);
    }

    velocity.integrate (false,true);
    velocity.distribute_local_to_global (rhs_correction.block(0));
    pressure.integrate (true,false);
    pressure.distribute_local_to_global (rhs_correction.block(1));
  }
  rhs_correction.compress(VectorOperation::add);

  LA::MPI::BlockVector stokes_rhs_correction (owned_partitioning, mpi_communicator);
  ChangeVectorTypes::copy(stokes_rhs_correction,rhs_correction);
  system_rhs.block(0) += stokes_rhs_correction.block(0);
  system_rhs.block(1) += stokes_rhs_correction.block(1);
}



template <int dim>
void StokesProblem<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "5.solve");
  Timer timer(mpi_communicator,true);

  timer.restart();
  // Below we define all the objects needed to build the GMG preconditioner:
  typedef dealii::LinearAlgebra::distributed::Vector<double> vector_t;

  // We choose a Chebyshev smoother, degree 4
  typedef PreconditionChebyshev<ABlockMatrixType,vector_t> SmootherType;
  mg::SmootherRelaxation<SmootherType, vector_t>
      mg_smoother;
  {
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
    {
      if (level > 0)
      {
        smoother_data[level].smoothing_range = 15.;
        smoother_data[level].degree = 4;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else
      {
        smoother_data[0].smoothing_range = 1e-3;
        smoother_data[0].degree = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
      }
      smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);
  }

  // Coarse Solver is just an application of the Chebyshev smoother setup
  // in such a way to be a solver
  MGCoarseGridApplySmoother<vector_t> mg_coarse;
  mg_coarse.initialize(mg_smoother);

  // Interface matrices
  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<ABlockMatrixType> > mg_interface_matrices;
  mg_interface_matrices.resize(0, triangulation.n_global_levels()-1);
  for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
    mg_interface_matrices[level].initialize(mg_matrices[level]);
  mg::Matrix<vector_t > mg_interface(mg_interface_matrices);

  // MG Matrix
  mg::Matrix<vector_t > mg_matrix(mg_matrices);

  // MG object
  Multigrid<vector_t > mg(mg_matrix,
                          mg_coarse,
                          mg_transfer,
                          mg_smoother,
                          mg_smoother);
  mg.set_edge_matrices(mg_interface, mg_interface);

  // GMG Preconditioner
  typedef PreconditionMG<dim, vector_t, MGTransferMatrixFree<dim,double> > APreconditioner;
  APreconditioner prec_A(dof_handler_v, mg, mg_transfer);

  // For the Mass matrix Preconditioner we choose a Chebyshev smoother setup
  // in a similar way to the coarse grid solver.
  typedef PreconditionChebyshev<MassMatrixType,vector_t> MassPreconditioner;
  MassPreconditioner prec_S;
  typename MassPreconditioner::AdditionalData prec_S_data;
  prec_S_data.smoothing_range = 1e-3;
  prec_S_data.degree = numbers::invalid_unsigned_int;
  prec_S_data.eig_cg_n_iterations = mass_matrix.m();
  prec_S_data.preconditioner = mass_matrix.get_matrix_diagonal_inverse();
  prec_S.initialize(mass_matrix,prec_S_data);

  using mp_inverse_t = LinearSolvers::InverseMatrix<MassMatrixType,MassPreconditioner>;
  const mp_inverse_t mp_inverse(mass_matrix, prec_S);

  const LinearSolvers::BlockDiagonalPreconditioner<APreconditioner,
      mp_inverse_t>
      preconditioner(prec_A, mp_inverse);

  LA::MPI::BlockVector distributed_solution(owned_partitioning,
                                            mpi_communicator);

  constraints.set_zero(distributed_solution);

  dealii::LinearAlgebra::distributed::BlockVector<double> solution_copy(2);
  dealii::LinearAlgebra::distributed::BlockVector<double> rhs_copy(2);

  stokes_matrix.initialize_dof_vector(solution_copy);
  stokes_matrix.initialize_dof_vector(rhs_copy);

  solution_copy.collect_sizes();
  rhs_copy.collect_sizes();

  ChangeVectorTypes::copy(solution_copy,distributed_solution);
  ChangeVectorTypes::copy(rhs_copy,system_rhs);

  SolverControl solver_control(100,
                               1e-10 * system_rhs.l2_norm());
  timer.stop();
  pcout << "   Setup GMG preconditioner timings:  " << timer.last_wall_time() << std::endl;


  {
    SolverFGMRES<dealii::LinearAlgebra::distributed::BlockVector<double>> solver(solver_control);

    solution_copy = 0;
    timer.restart();
    solver.solve(stokes_matrix,
                 solution_copy,
                 rhs_copy,
                 preconditioner);
    timer.stop();
    const double solve_time = timer.last_wall_time();
    unsigned int fgmres_m = solver_control.last_step();
    pcout << "   FGMRES Solve timings:              " << solve_time << "  (" << fgmres_m << " iterations)"
          << std::endl;
  }


  const unsigned int n_scalar = 1000;
  const unsigned int n_matvec = 100;
  const unsigned int n_prec = 10;
  const unsigned int n_mpi_stuff = 1000;

  LA::MPI::BlockVector tmp1, tmp2;
  tmp1.reinit(owned_partitioning, mpi_communicator);
  tmp2.reinit(owned_partitioning, mpi_communicator);
  tmp1 = system_rhs;
  tmp2 = system_rhs;

  dealii::LinearAlgebra::distributed::BlockVector<double> tmp3(2);
  dealii::LinearAlgebra::distributed::BlockVector<double> tmp4(2);
  stokes_matrix.initialize_dof_vector(tmp3);
  stokes_matrix.initialize_dof_vector(tmp4);
  tmp3.collect_sizes();
  tmp4.collect_sizes();
  ChangeVectorTypes::copy(tmp3,system_rhs);
  ChangeVectorTypes::copy(tmp4,system_rhs);

  pcout << std::endl;

  double dummy_val = 0.0;
  timer.restart();
  for (unsigned int i=0; i<n_scalar; ++i)
  {
    dummy_val += tmp1*tmp2;
  }
  timer.stop();
  const double scalar_tril = timer.last_wall_time()/n_scalar;

  pcout << "   Trilinos Scalar Product Timings: " << scalar_tril << std::endl;

  timer.restart();
  for (unsigned int i=0; i<n_scalar; ++i)
  {
    dummy_val += tmp3*tmp4;
  }
  timer.stop();
  const double scalar_deal = timer.last_wall_time()/n_scalar;

  pcout << "   deal.II Scalar Product Timings:  " << scalar_deal << std::endl;

  timer.restart();
  for (unsigned int i=0; i<n_matvec; ++i)
  {
    stokes_matrix.vmult(tmp3,tmp4);
    tmp4 += tmp3;
  }
  timer.stop();
  const double matvec = timer.last_wall_time()/n_matvec;

  pcout << "   Matrix-vector Product Timings:   " << matvec << std::endl;

  timer.restart();
  for (unsigned int i=0; i<n_prec; ++i)
  {
    preconditioner.vmult(tmp3,tmp4);
    tmp4 += tmp3;
  }
  timer.stop();
  const double prec = timer.last_wall_time()/n_prec;

  pcout << "   Preconditioner Vmult Timings:    " << prec << std::endl;

  timer.restart();
  for (unsigned int i=0; i<n_mpi_stuff; ++i)
  {
    dummy_val += Utilities::MPI::sum(i,MPI_COMM_WORLD);
  }
  timer.stop();
  const double mpisum = timer.last_wall_time()/n_mpi_stuff;

  pcout << "   MPI sum Timings:                 " << mpisum << std::endl;

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int total_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);
  int number = my_rank;
  MPI_Request request;
  timer.restart();
  for (unsigned int i=0; i<n_mpi_stuff; ++i)
  {
    if (total_ranks<=20)
      continue;
    for (int p=1; p<=20; ++p)
    {
      MPI_Isend(&number,1,MPI_INT,(my_rank+p)%total_ranks,p,MPI_COMM_WORLD,&request);
    }
    for (int p=1; p<=20; ++p)
    {
      MPI_Recv(&number,1,MPI_INT,MPI_ANY_SOURCE,p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  timer.stop();
  const double mpisend = timer.last_wall_time()/n_mpi_stuff;

  pcout << "   MPI send/receive Timings:        " << mpisend << std::endl;

  pcout << std::endl;


}



template <int dim>
void StokesProblem<dim>::refine_grid()
{
  TimerOutput::Scope t(computing_timer, "6.refine");

  triangulation.refine_global();
}



template <int dim>
void StokesProblem<dim>::run()
{
  const unsigned int n_cycles = 5;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
  {
    pcout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0)
      make_grid();
    else
      refine_grid();

    Timer timer(mpi_communicator,true);

    timer.restart();
    setup_system();
    timer.stop();
    pcout << std::endl
          << "   Setup DoFs timings:                 " << timer.last_wall_time() << std::endl;

    timer.restart();
    assemble_system();
    evaluate_viscosity();
    correct_stokes_rhs();
    timer.stop();
    pcout << "   Assemble System (RHS) timings:       " << timer.last_wall_time() << std::endl;

    solve();

    pcout << std::endl;
    computing_timer.print_summary();
    computing_timer.reset();
  }
}
} // namespace Step55



int main(int argc, char *argv[])
{
  unsigned int n_cycles = 5;
  if (argc>1)
    n_cycles = dealii::Utilities::string_to_int(argv[1]);

  try
  {
    using namespace dealii;
    using namespace Step55;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    StokesProblem<3> problem(2);
    problem.run(n_cycles);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
