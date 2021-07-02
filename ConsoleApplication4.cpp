#include<iostream>
#include <Eigen/core> 
#include <Eigen/Dense> 
#include<omp.h>
#include<vector>
#include<map>
    /* printf */
#include <math.h>  
#include<algorithm>
//#include <Eigen/LU>  
using namespace Eigen;
using namespace std;
#define MATRIX_SIZE 50

int main()
{
	double f = 0.153124;
	double x0 = -0.000002;
	double y0 = 0.000002;
	double fai = 0;
	double omi = 0;
	double ka = 0;
	double u = 0;
	double v = 0;
	double bx = 10000;
	VectorXd x1(20);
	VectorXd y1(20);
	VectorXd x2(20);
	VectorXd y2(20);
	x1 << -25.1960, -6.9443, 9.3021, 79.0605, 12.4810, 76.8791, 11.1227, 55.8956, 86.9755, 101.0204, -10.7387, 35.0030, 77.2341, 33.0724, 56.7053, 10.3672, 36.3343, 51.9777, 80.0891, 10.6496;//-25.1960, -6.9443, 9.3021, 79.0605, 12.4810, 76.8791, 11.1227, 55.8956, 86.9755, 101.0204, -10.7387, 35.0030, 77.2341, 33.0724, 56.7053, 10.3672, 36.3343, 51.9777, 80.0891, 10.6496;
	y1 << -94.0773, 60.8659, 62.4007, 59.1308, 37.3720, 28.9323, 19.0341, 14.1632, 14.3368, 13.6838, -8.7144, -6.8592, -10.8843, -30.4913, -30.3289, -59.7328, -59.0140, -64.2110, -60.5609, -74.0084;//-94.0773, 60.8659, 62.4007, 59.1308, 37.3720, 28.9323, 19.0341, 14.1632, 14.3368, 13.6838, -8.7144, -6.8592, -10.8843, -30.4913, -30.3289, -59.7328, -59.0140, -64.2110, -60.5609, -74.0084;
	x2 << -100.241, -85.6009, -69.4878, -0.4840, -66.1827, -1.2700, -67.0006, -21.7414, 9.0353, 23.2526, -87.4522, -41.9225, 0.2230, -42.8568, -19.3739, -65.0972, -39.3311, -23.8240, 4.4446, -64.5672;
	y2 << -96.7344, 59.0477, 60.9535, 59.2612, 35.9923, 29.04524, 17.6167, 13.7969, 14.6780, 14.3741, -10.6747, -7.7402, -10.7791, -31.4585, -30.7359, -61.3393, -59.9864, -64.8425, -60.4866, -75.6589;
	x1 = x1 / 1000;
	y1 = y1 / 1000;
	x2 = x2 / 1000;
	y2 = y2 / 1000;
	for (int i = 0; i < 20; i++)
	{
		x1[i] -= x0;
		y1[i] -= y0;
		x2[i] -= x0;
		y2[i] -= y0;
	}
	VectorXd X1 = x1;
	VectorXd Y1 = y1;
	VectorXd Z1 = y1;
	Z1.setConstant(-f);
	int size = y1.size();
	int i = 0;
	
	while (i >= 0)
	{
		double by = bx * u;
		double bz = bx * v;
		i++;
		double a1, a2, a3, b1, b2, b3, c1, c2, c3;
		a1 = cos(fai) * cos(ka) - sin(fai) * sin(omi) * sin(ka);
		if (abs(a1) == 0)
			a1 = 0;
		a2 = -cos(fai) * sin(ka) - sin(fai) * sin(omi) * cos(ka);
		if (abs(a2) == 0)
			a2 = 0;
		a3 = -sin(fai) * cos(omi);
		if (abs(a3) == 0)
			a3 = 0;

		b1 = cos(omi) * sin(ka);
		if (abs(b1) == 0)
			b1 = 0;
		b2 = cos(omi) * cos(ka);
		if (abs(b2) == 0)
			b2 = 0;
		b3 = -sin(omi);
		if (abs(b3) == 0)
			b3 = 0;

		c1 = sin(fai) * cos(ka) + cos(fai) * sin(omi) * sin(ka);
		if (abs(c1) == 0)
			c1 = 0;
		c2 = -sin(fai) * sin(ka) + cos(fai) * sin(omi) * cos(ka);
		if (abs(c2) == 0)
			c2 = 0;
		c3 = cos(fai) * cos(omi);
		if (abs(c3) == 0)
			c3 = 0;
		Matrix3d R2;
		R2 << a1, a2, a3,
			b1, b2, b3,
			c1, c2, c3;
		MatrixXd Xzhuan;
		MatrixXd XX2(3, size);
		XX2 << x2.transpose(),
			y2.transpose(),
			Z1.transpose();
		//cout << x2.transpose() << endl;
		//cout << y2.transpose() << endl;
		Xzhuan = R2 * XX2;
		//cout << Xzhuan << endl;
		VectorXd N1(size);
		VectorXd N2(size);
		for (int g = 0; g < size; g++)
		{
			N1[g] = (bx * Xzhuan(2, g) - bz * Xzhuan(0, g)) / (X1(g) * Xzhuan(2, g) - Xzhuan(0, g) * Z1(g));
			N2[g] = (bx * Z1(g) - bz * X1(g)) / (X1(g) * Xzhuan(2, g) - Xzhuan(0, g) * Z1(g));
		}
		VectorXd Q(size);
		for (int p = 0; p < size; p++)
		{
			double Qn = N1(p) * Y1(p) - N2(p) * Xzhuan(1, p) - by;
			Q[p] = Qn;
			//std::cout << Qn << endl;
		}
		VectorXd L = Q;
		MatrixXd a11(5, size);
		for (int w = 0; w < size; w++)
		{
			a11(0, w) = -Xzhuan(0, w) * Xzhuan(1, w) * N2(w) / Xzhuan(2, w);
			a11(1, w) = -(Xzhuan(2, w) + pow(Xzhuan(1, w),2) / Xzhuan(2, w)) * N2(w);
			a11(2, w) = Xzhuan(0, w) * N2(w);
			a11(3, w) = bx;
			a11(4, w) = -bx * Xzhuan(1, w) / Xzhuan(2, w);
		}
		MatrixXd A = a11.transpose();
		MatrixXd Qxx = a11 * A;
		VectorXd Xc = Qxx.inverse() * a11 *L;
		double dfai = Xc(0);
		double domi = Xc(1);
		double dkap = Xc(2);
		double du = Xc(3);
		double dv = Xc(4);
		fai = fai + dfai;
		omi = omi + domi;
		ka = ka + dkap;
		u = u + du;
		v = v + dv;
		VectorXd Xc_result = Xc.transpose();
		for (int o = 0; o < Xc_result.size(); o++)
		{
			if (Xc_result[o] > 0.00003)
				goto re;

		}
		goto end;
	re:;
	}
end:;
	double pi = 3.1415926;
	cout <<"fai:"<< fai * 360 / (2 * pi) << endl;
	cout <<"omi:"<< omi * 360 / (2 * pi) << endl;
	cout <<"ka:"<< ka * 360 / (2 * pi) << endl;
	cout <<"u:"<< u * 360 / (2 * pi) << endl;
	cout <<"v:"<< v * 360 / (2 * pi) << endl;
	cout <<"bx:"<< bx << endl;

};