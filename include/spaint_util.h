#pragma once 

#include <vector>
#include <limits>
#include <cmath>

#include "spaint.h"
#include "mat3l.h"
#include "vec2.h"

// Utility functions for painter
namespace spaint {
	
	// Render hermite spline for given points
	void hermite_spline(spaint::painter& p, const std::vector<cppmath::vec2>& points, int steps = 1000) {
		if (points.size() < 2)
			return;
		
		if (steps <= 0)
			return;
		
		double d = 1.0 / (double) steps;
		
		if (d == 0.0)
			return;
		
		auto P = [](double t, double p0, double p1, double m0, double m1) -> double { 
			double t2 = t * t;
			double t3 = t2 * t;
			return  (2.0 * t3 - 3.0 * t2 + 1.0) * p0
					+
					(t3 - 2.0 * t2 + t) * m0
					+
					(-2.0 * t3 + 3.0 * t2) * p1
					+
					(t3 - t2) * m1;
		};
		
		std::vector<cppmath::vec2> derivatives;
		for (int i = 0; i < points.size(); ++i) {
			cppmath::vec2 ptan;
			
			if (i == 0) 
				ptan = points[1] - points[0];
			else if (i == points.size() - 1)
				ptan = points[i] - points[i - 1];
			else
				ptan = (points[i + 1] - points[i - 1]) / 2.0;
				
			derivatives.push_back(ptan);
		}
		
		for (int i = 0; i < points.size() - 1; ++i) {
			double p_x = P(0.0, points[i].x, points[i + 1].x, derivatives[i].x, derivatives[i + 1].x);
			double p_y = P(0.0, points[i].y, points[i + 1].y, derivatives[i].y, derivatives[i + 1].y);
			
			for (double t = d; t <= 1.0; t += d) {
				double x = P(t, points[i].x, points[i + 1].x, derivatives[i].x, derivatives[i + 1].x);
				double y = P(t, points[i].y, points[i + 1].y, derivatives[i].y, derivatives[i + 1].y);
				
				p.line(p_x, p_y, x, y);
				
				p_x = x;
				p_y = y;
			}
		}
	};
	
	// Render hermite spline for given points and given derivatives
	void hermite_spline(spaint::painter& p, const std::vector<cppmath::vec2>& points, const std::vector<cppmath::vec2>& derivatives, int steps = 1000) {
		if (points.size() < 2)
			return;
		
		if (points.size() > derivatives.size())
			return;
		
		if (steps <= 0)
			return;
		
		double d = 1.0 / (double) steps;
		
		if (d == 0.0)
			return;
		
		auto P = [](double t, double p0, double p1, double m0, double m1) -> double { 
			double t2 = t * t;
			double t3 = t2 * t;
			return  (2.0 * t3 - 3.0 * t2 + 1.0) * p0
					+
					(t3 - 2.0 * t2 + t) * m0
					+
					(-2.0 * t3 + 3.0 * t2) * p1
					+
					(t3 - t2) * m1;
		};
		
		for (int i = 0; i < points.size() - 1; ++i) {
			double p_x = P(0.0, points[i].x, points[i + 1].x, derivatives[i].x, derivatives[i + 1].x);
			double p_y = P(0.0, points[i].y, points[i + 1].y, derivatives[i].y, derivatives[i + 1].y);
			
			for (double t = d; t <= 1.0; t += d) {
				double x = P(t, points[i].x, points[i + 1].x, derivatives[i].x, derivatives[i + 1].x);
				double y = P(t, points[i].y, points[i + 1].y, derivatives[i].y, derivatives[i + 1].y);
				
				p.line(p_x, p_y, x, y);
				
				p_x = x;
				p_y = y;
			}
		}
	};
	
	std::vector<cppmath::vec2> hermite_derivatives(const std::vector<cppmath::vec2>& points) {
		std::vector<cppmath::vec2> derivatives;
		for (int i = 0; i < points.size(); ++i) {
			cppmath::vec2 ptan;
			
			if (i == 0) 
				ptan = points[1] - points[0];
			else if (i == points.size() - 1)
				ptan = points[i] - points[i - 1];
			else
				ptan = (points[i + 1] - points[i - 1]) / 2.0;
				
			derivatives.push_back(ptan);
		}
		
		return derivatives;
	};
	
	// Calculates curve derivative vector on given sector [Xi, Xi+1] at given point t.
	cppmath::vec2 hermite_derivative(const std::vector<cppmath::vec2>& points, const std::vector<cppmath::vec2>& derivatives, int sector, double t) {
		if (sector < 0 || sector >= points.size())
			return std::numeric_limits<double>::quiet_NaN();
		
		auto P = [](double t, double p0, double p1, double m0, double m1) -> double { 
			double t2 = t * t;
			double t3 = t2 * t;
			return  m0 * (3.0 * t2 - 4.0 * t + 1.0)
					+ 
					t * (m1 * (3.0 * t - 2.0) 
						+ 
						6.0 * p0 * t 
						- 
						6.0 * p0 
						- 
						6.0 * p1 * t 
						+ 
						6.0 * p1);
		};
		
		return cppmath::vec2(P(t, points[sector].x, points[sector + 1].x, derivatives[sector].x, derivatives[sector + 1].x),
							 P(t, points[sector].y, points[sector + 1].y, derivatives[sector].y, derivatives[sector + 1].y));			
	};
	
	// Calculates curve point vector on given sector [Xi, Xi+1] at given point t.
	cppmath::vec2 hermite_point(const std::vector<cppmath::vec2>& points, const std::vector<cppmath::vec2>& derivatives, int sector, double t) {
		if (sector < 0 || sector >= points.size())
			return std::numeric_limits<double>::quiet_NaN();
		
		auto P = [](double t, double p0, double p1, double m0, double m1) -> double { 
			double t2 = t * t;
			double t3 = t2 * t;
			return  (2.0 * t3 - 3.0 * t2 + 1.0) * p0
					+
					(t3 - 2.0 * t2 + t) * m0
					+
					(-2.0 * t3 + 3.0 * t2) * p1
					+
					(t3 - t2) * m1;
		};
		
		return cppmath::vec2(P(t, points[sector].x, points[sector + 1].x, derivatives[sector].x, derivatives[sector + 1].x),
							 P(t, points[sector].y, points[sector + 1].y, derivatives[sector].y, derivatives[sector + 1].y));			
	};

	void bezier_curve(spaint::painter& p, const std::vector<cppmath::vec2>& points, int steps = 1000) {
		if (points.size() > 1) {
			
			if (steps <= 0)
				return;
		
			double d = 1.0 / (double) steps;
			
			if (d == 0)
				return;
			
			std::function<double(int)> factorial;
			factorial = [&factorial](int n) -> double {
				if (n <= 1)
					return 1;
				return factorial(n - 1) * n;
			};
			
			std::vector<double> C;
			for (int i = 0; i < points.size(); ++i) 
				C.push_back(factorial(points.size() - 1) / (factorial(i) * factorial(points.size() - 1 - i)));
			
			auto B = [&C](int i, int n, double t) -> double { 
				return C[i] * std::pow(t, i) * std::pow(1 - t, n - i);
			};
			
			// Bezier:
			// B[i, n](t) = C[n, i] * t ^ i * (1 - t) ^ (n - i)
			// r(t) = SUM[ i = 0; i <= N; B[i, n](t) * P[i]
			
			cppmath::vec2 point_old = points[0];
			
			for (double t = 0.0; t <= 1.0; t += d) {
				cppmath::vec2 point;
				
				for (int i = 0; i < points.size(); ++i) {
					double part_mult = B(i, points.size() - 1, t);
					point += part_mult * points[i];
				}
				
				if (t != 0.0) 
					p.line(point_old.x, point_old.y, point.x, point.y);
				point_old = point;
			}				
		}
	};
	
	void rational_bezier(spaint::painter& p, const std::vector<cppmath::vec2>& points, const std::vector<double>& weights, int steps = 1000) {
		if (points.size() > 1) {
			
			if (steps <= 0)
				return;
		
			double d = 1.0 / (double) steps;
			
			if (d == 0)
				return;
			
			std::function<double(int)> factorial;
			factorial = [&factorial](int n) -> double {
				if (n <= 1)
					return 1;
				return factorial(n - 1) * n;
			};
			
			std::vector<double> C;
			for (int i = 0; i < points.size(); ++i) 
				C.push_back(factorial(points.size() - 1) / (factorial(i) * factorial(points.size() - 1 - i)));
			
			auto B = [&C, &weights](int i, int n, double t) -> double { 
				return C[i] * std::pow(t, i) * std::pow(1 - t, n - i);
			};
			
			auto W = [&B, &points, &weights](double t) -> double {
				double wUuu = 0.0;
				for (int i = 0; i < points.size(); ++i)
					wUuu += B(i, points.size() - 1, t) * weights[i];
				
				return wUuu;
			};
			
			// Bezier:
			// B[i, n](t) = C[n, i] * t ^ i * (1 - t) ^ (n - i)
			// r(t) = SUM[ i = 0; i <= N; B[i, n](t) * P[i]
			
			cppmath::vec2 point_old = points[0];
			
			for (double t = 0.0; t <= 1.0; t += d) {
				cppmath::vec2 point;
				
				for (int i = 0; i < points.size(); ++i) {
					double part_mult = B(i, points.size() - 1, t) * weights[i] / W(t);
					point += part_mult * points[i];
				}
				
				if (t != 0.0) 
					p.line(point_old.x, point_old.y, point.x, point.y);
				point_old = point;
			}				
		}
	};

	void cubic_spline(spaint::painter& p, const std::vector<cppmath::vec2>& points, int steps = 1000, double start_slope = std::numeric_limits<double>::quiet_NaN(), double end_slope = std::numeric_limits<double>::quiet_NaN()) {
		
		// https://www.codeproject.com/articles/560163/csharp-cubic-spline-interpolation
		
		if (points.size() > 1) {
			
			using namespace std;
			
			if (steps <= 0)
				return;
		
			double d = 1.0 / (double) steps;
			
			if (d == 0)
				return;
			
			// Assume vector is sorted
			
			// Calculate
			int n = points.size();
			vector<double> r(n);
			
			cppmath::mat3l m(n);
			double dx1, dx2, dy1, dy2;
			
			if (std::isnan(start_slope)) {
				dx1 = points[1].x - points[0].x;
				m.C[0] = 1.0 / dx1;
				m.B[0] = 2.0 * m.C[0];
				r[0] = 3.0 * (points[1].y - points[0].y) / (dx1 * dx1);
			} else {
				m.B[0] = 1;
				r[0] = start_slope;
			}
			
			for (int i = 1; i < n - 1; ++i) {
				dx1 = points[i].x - points[i - 1].x;
				dx2 = points[i + 1].x - points[i].x;
				
				m.A[i] = 1.0 / dx1;
				m.C[i] = 1.0 / dx2;
				m.B[i] = 2.0 * (m.A[i] + m.C[i]);
				
				dy1 = points[i].y - points[i - 1].y;
				dy2 = points[i + 1].y - points[i].y;
				r[i] = 3.0 * (dy1 / (dx1 * dx1) + dy2 / (dx2 * dx2));
			}
			
			if (std::isnan(end_slope)) {
				dx1 = points[n - 1].x - points[n - 2].x;
				dy1 = points[n - 1].y - points[n - 2].y;
				m.A[n - 1] = 1.0 / dx1;
				m.B[n - 1] = 2.0 * m.A[n - 1];
				r[n - 1] = 3.0 * (dy1 / (dx1 * dx1));
			} else {
				m.B[n - 1] = 1;
				r[n - 1] = end_slope;
			}
			
			vector<double> k = m.solve(r);
			
			vector<double> a(n - 1);
			vector<double> b(n - 1);
			
			for (int i = 1; i < n; ++i) {
				dx1 = points[i].x - points[i - 1].x;
				dy1 = points[i].y - points[i - 1].y;
				a[i - 1] = k[i - 1] * dx1 - dy1;
				b[i - 1] = -k[i] * dx1 + dy1;
			}
			
			// Render
			
			int lastIndex = 0;
			double distance = points.back().x - points.front().x;
			cppmath::vec2 point_old = points[0];
			
			
			for (int i = 0; i < points.size() - 1; ++i) {
				for (double t = 0.0; t <= 1.0; t += d) {
					cppmath::vec2 point;
					
					point.x = points[i].x + t * (points[i + 1].x - points[i].x);
					point.y = (1 - t) * points[i].y + t * points[i + 1].y + t * (1 - t) * (a[i] * (1 - t) + b[i] * t);
					
					if (t != 0.0) 
						p.line(point_old.x, point_old.y, point.x, point.y);
					point_old = point;
				}	
			}
		}
	};
	
	void lagrange_spline(spaint::painter& p, const std::vector<cppmath::vec2>& points, int steps = 1000, double start = std::numeric_limits<double>::quiet_NaN(), double end = std::numeric_limits<double>::quiet_NaN()) {
		if (points.size() > 1) {
			
			if (steps <= 0)
				return;
		
			double d = 1.0 / (double) steps;
			
			if (d == 0)
				return;
			
			auto interpLagPol = [&points](double x) -> double { 
				double pol = 0.0;
				
				for (int i = 0; i < points.size(); ++i) {
					double basic_pol = 1;
					
					for (int j = 0; j < points.size(); ++j)
						if (i != j)
							basic_pol *= (x - points[j].x) / (points[i].x - points[j].x);
						
					pol += basic_pol * points[i].y;
				}
				
				return pol;
			};
			
			// Assume vector is sorted
			
			if (std::isnan(start))
				start = points.front().x;
			
			if (std::isnan(end))
				end = points.back().x;
			
			
			cppmath::vec2 point_old = start;
			
			for (double t = 0; t <= 1.0; t += d) {
				cppmath::vec2 point;
				
				point.x = (end - start) * t + start;
				point.y = interpLagPol(point.x);
				
				if (t != 0.0) 
					p.line(point_old.x, point_old.y, point.x, point.y);
				point_old = point;
			}				
		}
	};
	
	/*void b_spline(spaint::painter& p, const std::vector<cppmath::vec2>& points, int steps = 1000) {
		if (points.size() > 1) {
			
			if (steps <= 0)
				return;
		
			double d = 1.0 / (double) steps;
			
			if (d == 0)
				return;
			void BSplineCurve(const Dot &point1, 
                  const Dot &point2, 
                  const Dot &point3,
                  const Dot &point4, 
                  const double t,
                  Dot &result)
			{
				const double t2 = t * t;
				const double t3 = t2 * t;
				const double mt = 1.0 - t;
				const double mt3 = mt * mt * mt;

				const double bi3 = mt3;
				const double bi2 = 3 * t3 - 6 * t2 + 4;
				const double bi1 =-3 * t3 + 3 * t2 + 3 * t + 1;
				const double bi  = t3;

				result.x = point1.x * bi3 + 
						   point2.x * bi2 +
						   point3.x * bi1 +
						   point4.x * bi;
				result.x /= 6.0;

				result.y = point1.y * bi3 + 
						   point2.y * bi2 +
						   point3.y * bi1 +
						   point4.y * bi;
				result.y /= 6.0;
			}
			
			auto B = [&points](double t, int i) -> cppmath::vec2 {
				double t2 = t * t;
				double t3 = t2 * t;
				double mt = 1.0 - t;
				double mt3 = mt * mt * mt;
				
				double bi3 = mt3;
				double bi2 = 3.0 * t3 - 6.0 * t2 + 4.0;
				double bi1 =-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0;
				double bi  = t3;
				
				return cppmath::vec2(
			};
			
			
			cppmath::vec2 point_old = start;
			
			for (double t = 0; t <= 1.0; t += d) {
				cppmath::vec2 point;
				
				point.x = (end - start) * t + start;
				point.y = interpLagPol(point.x);
				
				if (t != 0.0) 
					p.line(point_old.x, point_old.y, point.x, point.y);
				point_old = point;
			}				
		}
	};*/
};