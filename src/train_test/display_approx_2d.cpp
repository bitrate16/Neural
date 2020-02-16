/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 * Utility for rendering approx 2D network on given interval
 * Arguments:
 *  --start=%     Interval start
 *  --end=%       Interval end
 *  --function=%  If exists, will render function in background
 *  --network=%   Network to display
 *  --display_off=%  Offset value for displaying plot
 *  --display_offv=% Vertival offset for displaying plot
 * 
 * Make:
 * g++ src/train_test/display_approx_2d.cpp -o bin/display_approx_2d -O3 --std=c++17 -Iinclude -lstdc++fs -lstdc++ -lm -lX11
 *
 * Example:
 * ./bin/display_approx_2d --function="sin(t*3.14*2.0)*0.5+0.5" --network=networks/approx_sin.neetwook --display_off=0.1 --display_offv=0.1
 * 
 */

#include <iostream>
#include <unistd.h>
#include <fstream>
#include <cmath>

#include "MultiLayerNetwork.h"
#include "math_func_util.h"
#include "math_func.h"
#include "pargs.h"

#include "spaint.h"
#include "Color.h"

using namespace spaint;

#define KEY_ESCAPE 9
#define KEY_R      27


class scene : public component {
	
public:

	NNSpace::MLNet net;
	std::string net_path;
	double start = 0.0;
	double end   = 1.0;
	double off   = 0.0;
	double offv  = 0.0;
	
	std::map<std::string, double> values;
	std::map<std::string, std::function<double(const std::vector<double>&)>> functions;
	math_func::func* back_func = nullptr;
	
	// Generated on resize, point set to render graph
	std::vector<double> point_set;
	
	bool mouse_down = 0;
	bool resized    = 0;
	bool updated    = 0;
	
	
	void create() {
		get_paint().init_font();
		get_window().set_title("Approx plot");
		
		updated = 1;
		get_plot_set();
		
		// Init default functions
		functions["sin"] = [](std::vector<double> t) {
			return std::sin(t.size() == 0 ? 0.0 : t[0]);
		};
		functions["cos"] = [](std::vector<double> t) {
			return std::cos(t.size() == 0 ? 0.0 : t[0]);
		};
		functions["tan"] = [](std::vector<double> t) {
			return std::tan(t.size() == 0 ? 0.0 : t[0]);
		};
		functions["ctan"] = [](std::vector<double> t) {
			return std::cos(t.size() == 0 ? 0.0 : t[0]) / std::sin(t.size() == 0 ? 0.0 : t[0]);
		};
		functions["exp"] = [](std::vector<double> t) {
			return std::exp(t.size() == 0 ? 0.0 : t[0]);
		};
		functions["pow"] = [](std::vector<double> t) {
			if (t.size() == 0)
				return 0.0;
			if (t.size() == 1)
				return 1.0;
			return std::pow(t[0], t[1]);
		};
		functions["log"] = [](std::vector<double> t) {
			if (t.size() == 0)
				return 0.0;
			return std::log(t[0]);
		};
		functions["arcsin"] = [](std::vector<double> t) {
			return std::asin(t.size() == 0 ? 0.0 : t[0]);
		};
		functions["arccos"] = [](std::vector<double> t) {
			return std::acos(t.size() == 0 ? 0.0 : t[0]);
		};
		functions["arctan"] = [](std::vector<double> t) {
			return std::atan(t.size() == 0 ? 0.0 : t[0]);
		};
		
		reload();
	};
	
	void destroy() {};
	
	// Set of points used for displaying
	void get_plot_set() {
		point_set.resize(get_window().get_width() + 2, 0.0);
		
		double interval = (end - start) / (1.0 - 2.0 * off);
		double step = interval / static_cast<double>(get_window().get_width() + 2);
		int i = 0;
		for (double d = start - interval * off + step; d < end + interval * off - step; d += step)
			point_set[i++] = d;
	};
	
	// Does reloading of the network from path
	bool reload() {		
		std::ifstream ifs;
		ifs.open(net_path);
		if (ifs.fail()) {
			std::cout << "File " << net_path << " not found" << std::endl;
			return 0;
		}
		
		if (!net.deserialize(ifs)) {
			std::cout << "Deserialize failed" << std::endl;
			ifs.close();
			return 0;
		}
		ifs.close();
		
		return 1;
	};
	
	void resize() {
		resized = 1;
		get_plot_set();
	};
	
	void loop() {
		window& w = get_window();
		painter& p = w.get_paint();
		
		// Block untill event is reached
		if (!mouse_down) w.wait_event(1);
		
		if (w.has_key_event(0))
			if (w.get_key_down() == KEY_ESCAPE)
				w.stop();
			if (w.get_key_down() == KEY_R) 
				updated = 1;
		
		if (w.has_mouse_event(0)) 
			if (w.get_button_down() == Button1) 
				mouse_down = 1;
			else if (w.get_button_up() == Button1) 
				mouse_down = 0;
			
		w.clear_events();
			
		if (resized || updated) {
			
			// Reload on R key
			if (updated)
				if (!reload())
					exit(0);
			
			resized = 0;
			updated = 0;
			
			p.clear();
			
			// Calculate step intervals
			double interval = (start - end) / (1.0 - 2.0 * off);
			double vinterval = 1.0 / (1.0 - 2.0 * offv);
			
			// Render offset borders
			p.color(0, 0, 255);
			p.line(get_window().get_width() * off, 0, get_window().get_width() * off, get_window().get_height());
			p.line(get_window().get_width() * (1.0 - off), 0, get_window().get_width() * (1.0 - off), get_window().get_height());
			p.line(0, get_window().get_height() / 2, get_window().get_width(), get_window().get_height() / 2);
			p.line(0, get_window().get_height() * offv, get_window().get_width(), get_window().get_height() * offv);
			p.line(0, get_window().get_height() * (1.0 - offv), get_window().get_width(), get_window().get_height() * (1.0 - offv));
			
			std::vector<double> input(1);
			std::vector<double> output(1);
			
			for (int i = 0; i < point_set.size(); ++i) {
				input[0] = point_set[i];
				output   = net.run(input);
				
				int x = get_window().get_width() * ((point_set[0] - point_set[i]) / interval);
				int y;
				
				if (back_func) {
					
					// Render background function
					p.color(255, 255, 0);
					values["t"] = point_set[i];
					double e = back_func->evaluate(values, functions);
					y = get_window().get_height() - get_window().get_height() * ((e + 1.0) * (0.5 - offv) + offv);
					p.point(x, y);
					
					// Render error value
					p.color(255, 0, 0);
					y = get_window().get_height() - get_window().get_height() * ((std::abs(output[0] - e) + 1.0) * (0.5 - offv) + offv);
					p.point(x, y);
				}
				
				// Render network
				p.color(0, 255, 0);
				y = get_window().get_height() - get_window().get_height() * ((output[0] + 1.0) * (0.5 - offv) + offv);
				p.point(x, y);
			}
			
		} else {
			// ...
		}
	};
};


int main(int argc, const char** argv) {
	pargs::pargs args(argc, argv);
	
	double start         = args["--start"]    && args["--start"]->is_real()      ? args["--start"]->real()      : 0.0;
	double end           = args["--end"]      && args["--end"]->is_real()        ? args["--end"]->real()        : 1.0;
	std::string function = args["--function"] && args["--function"]->is_string() ? args["--function"]->string() : "";
	std::string network  = args["--network"]  && args["--network"]->is_string()  ? args["--network"]->string()  : "network.neetwook";
	double off           = args["--display_off"]  && args["--display_off"]->is_real()  ? args["--display_off"]->real()  : 0.0;
	double offv          = args["--display_offv"] && args["--display_offv"]->is_real() ? args["--display_offv"]->real() : 0.0;
	
	scene s;
	s.net_path = network;
	s.start = start;
	s.end   = end;
	s.off  = off;
	s.offv = offv;
	
	if (function != "") {
		math_func::func* func = math_func::parse(function);
		if (!func) {
			std::cout << "Error parsing function" << std::endl;
			return 0;
		}
		
		s.back_func = math_func::optimize(func);
		delete func;
	}
	
	window w(&s, 400, 200, 0);
	w.start();
	return 0;
};

