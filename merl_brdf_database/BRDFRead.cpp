// Copyright 2005 Mitsubishi Electric Research Laboratories All Rights Reserved.

// Permission to use, copy and modify this software and its documentation without
// fee for educational, research and non-profit purposes, is hereby granted, provided
// that the above copyright notice and the following three paragraphs appear in all copies.

// To request permission to incorporate this software into commercial products contact:
// Vice President of Marketing and Business Development;
// Mitsubishi Electric Research Laboratories (MERL), 201 Broadway, Cambridge, MA 02139 or 
// <license@merl.com>.

// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED
// HEREUNDER IS ON AN "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
// UPDATES, ENHANCEMENTS OR MODIFICATIONS.

#include "boost/numpy.hpp"
namespace py = boost::python;
namespace np = boost::numpy;
////

#include "stdlib.h"
#include "math.h"

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360

#define RED_SCALE (1.0/1500.0)
#define GREEN_SCALE (1.15/1500.0)
#define BLUE_SCALE (1.66/1500.0)
#define M_PI	3.1415926535897932384626433832795

// cross product of two vectors
void cross_product (double* v1, double* v2, double* out)
{
	out[0] = v1[1]*v2[2] - v1[2]*v2[1];
	out[1] = v1[2]*v2[0] - v1[0]*v2[2];
	out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

// normalize vector
void normalize(double* v)
{
	// normalize
	double len = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	v[0] = v[0] / len;
	v[1] = v[1] / len;
	v[2] = v[2] / len;
}

// rotate vector along one axis
void rotate_vector(double* vector, double* axis, double angle, double* out)
{
	double temp;
	double cross[3];
	double cos_ang = cos(angle);
	double sin_ang = sin(angle);

	out[0] = vector[0] * cos_ang;
	out[1] = vector[1] * cos_ang;
	out[2] = vector[2] * cos_ang;

	temp = axis[0]*vector[0]+axis[1]*vector[1]+axis[2]*vector[2];
	temp = temp*(1.0-cos_ang);

	out[0] += axis[0] * temp;
	out[1] += axis[1] * temp;
	out[2] += axis[2] * temp;

	cross_product (axis,vector,cross);
	
	out[0] += cross[0] * sin_ang;
	out[1] += cross[1] * sin_ang;
	out[2] += cross[2] * sin_ang;
}


// convert standard coordinates to half vector/difference vector coordinates
void std_coords_to_half_diff_coords(double theta_in, double fi_in, double theta_out, double fi_out,
								double& theta_half,double& fi_half,double& theta_diff,double& fi_diff )
{

	// compute in vector
	double in_vec_z = cos(theta_in);
	double proj_in_vec = sin(theta_in);
	double in_vec_x = proj_in_vec*cos(fi_in);
	double in_vec_y = proj_in_vec*sin(fi_in);
	double in[3]= {in_vec_x,in_vec_y,in_vec_z};
	normalize(in);


	// compute out vector
	double out_vec_z = cos(theta_out);
	double proj_out_vec = sin(theta_out);
	double out_vec_x = proj_out_vec*cos(fi_out);
	double out_vec_y = proj_out_vec*sin(fi_out);
	double out[3]= {out_vec_x,out_vec_y,out_vec_z};
	normalize(out);


	// compute halfway vector
	double half_x = (in_vec_x + out_vec_x)/2.0f;
	double half_y = (in_vec_y + out_vec_y)/2.0f;
	double half_z = (in_vec_z + out_vec_z)/2.0f;
	double half[3] = {half_x,half_y,half_z};
	normalize(half);

	// compute  theta_half, fi_half
	theta_half = acos(half[2]);
	fi_half = atan2(half[1], half[0]);


	double bi_normal[3] = {0.0, 1.0, 0.0};
	double normal[3] = { 0.0, 0.0, 1.0 };
	double temp[3];
	double diff[3];

	// compute diff vector
	rotate_vector(in, normal , -fi_half, temp);
	rotate_vector(temp, bi_normal, -theta_half, diff);
	
	// compute  theta_diff, fi_diff	
	theta_diff = acos(diff[2]);
	fi_diff = atan2(diff[1], diff[0]);

}


// Lookup theta_half index
// This is a non-linear mapping!
// In:  [0 .. pi/2]
// Out: [0 .. 89]
inline int theta_half_index(double theta_half)
{
	if (theta_half <= 0.0)
		return 0;
	double theta_half_deg = ((theta_half / (M_PI/2.0))*BRDF_SAMPLING_RES_THETA_H);
	double temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H;
	temp = sqrt(temp);
	int ret_val = (int)temp;
	if (ret_val < 0) ret_val = 0;
	if (ret_val >= BRDF_SAMPLING_RES_THETA_H)
		ret_val = BRDF_SAMPLING_RES_THETA_H-1;
	return ret_val;
}


// Lookup theta_diff index
// In:  [0 .. pi/2]
// Out: [0 .. 89]
inline int theta_diff_index(double theta_diff)
{
	int tmp = int(theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D);
	if (tmp < 0)
		return 0;
	else if (tmp < BRDF_SAMPLING_RES_THETA_D - 1)
		return tmp;
	else
		return BRDF_SAMPLING_RES_THETA_D - 1;
}


// Lookup phi_diff index
inline int phi_diff_index(double phi_diff)
{
	// Because of reciprocity, the BRDF is unchanged under
	// phi_diff -> phi_diff + M_PI
	if (phi_diff < 0.0)
		phi_diff += M_PI;

	// In: phi_diff in [0 .. pi]
	// Out: tmp in [0 .. 179]
	int tmp = int(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2);
	if (tmp < 0)	
		return 0;
	else if (tmp < BRDF_SAMPLING_RES_PHI_D / 2 - 1)
		return tmp;
	else
		return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
}


// Given a pair of incoming/outgoing angles, look up the BRDF.
void lookup_brdf_val(double* brdf, double theta_in, double fi_in,
			  double theta_out, double fi_out, 
			  double& red_val,double& green_val,double& blue_val)
{
	// Convert to halfangle / difference angle coordinates
	double theta_half, fi_half, theta_diff, fi_diff;
	
	std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out,
		       theta_half, fi_half, theta_diff, fi_diff);


	// Find index.
	// Note that phi_half is ignored, since isotropic BRDFs are assumed
	int ind = phi_diff_index(fi_diff) +
		  theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
		  theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 *
					         BRDF_SAMPLING_RES_THETA_D;

	red_val = brdf[ind] * RED_SCALE;
	green_val = brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2] * GREEN_SCALE;
	blue_val = brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D] * BLUE_SCALE;


	if (red_val < 0.0 || green_val < 0.0 || blue_val < 0.0)
		fprintf(stderr, "Below horizon.\n");

}


// Read BRDF data
int read_brdf(const char *filename, double* &brdf)
{
	FILE *f = fopen(filename, "rb");
	if (!f)
		return false;

	int dims[3];
	fread(dims, sizeof(int), 3, f);
	int n = dims[0] * dims[1] * dims[2];
	if (n != BRDF_SAMPLING_RES_THETA_H *
		 BRDF_SAMPLING_RES_THETA_D *
		 BRDF_SAMPLING_RES_PHI_D / 2) 
	{
		fprintf(stderr, "Dimensions don't match\n");
		fclose(f);
		return 0;
	}

	brdf = (double*) malloc (sizeof(double)*3*n);
	fread(brdf, sizeof(double), 3*n, f);

	fclose(f);
	return 3*n;
}


////////////////////////////////////////////////////////
//Add bellow functions for Boost.python
////////////////////////////////////////////////////////
py::tuple lookup_brdf_valpy(np::ndarray brdf,  double theta_in, double fi_in, double theta_out, double fi_out){
	// Convert to halfangle / difference angle coordinates
	double theta_half, fi_half, theta_diff, fi_diff;

	std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out, theta_half, fi_half, theta_diff, fi_diff);

	// Find index.
	// Note that phi_half is ignored, since isotropic BRDFs are assumed
	int ind = phi_diff_index(fi_diff) +
		  theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
		  theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 *
					         BRDF_SAMPLING_RES_THETA_D;

	double red_val = py::extract<double>(brdf[ind]) * RED_SCALE;
	double green_val = py::extract<double>(brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2]) * GREEN_SCALE;
	double blue_val = py::extract<double>(brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D]) * BLUE_SCALE;

	if (red_val < 0.0 || green_val < 0.0 || blue_val < 0.0){
		fprintf(stderr, "Below horizon.\n");
	}
	return py::make_tuple(red_val, green_val, blue_val);
}

// Read BRDF data
np::ndarray read_brdfpy(const char *filename)
{
	double* brdf;
	int dim = read_brdf(filename, brdf);

	if (dim == 0)
	{
		py::tuple shape = py::make_tuple(0);
        np::dtype dtype = np::dtype::get_builtin<double>();
        np::ndarray brdf_mat = np::zeros(shape, dtype);
        return brdf_mat;
	}

	py::tuple shape = py::make_tuple(dim);
    np::dtype dtype = np::dtype::get_builtin<double>();
    np::ndarray brdf_mat = np::zeros(shape, dtype);

    for(int i = 0; i < dim; i++ ){
        brdf_mat[i] = brdf[i];
    }

	return brdf_mat;
}



/////////////////////////////////////////////////////////////////

BOOST_PYTHON_MODULE(BRDFRead) {
  Py_Initialize();
  np::initialize();
  py::def("read_brdf", read_brdfpy);
  py::def("lookup_brdf_val", lookup_brdf_valpy);
}
