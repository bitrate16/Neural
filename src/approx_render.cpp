/*
    Example shows use of spaint and cppmath::math::right_turn
	
	cpp math utilities
    Copyright (C) 2019-3041  bitrate16

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <fstream>
#include <cmath>
#include <unistd.h>

#include "Network.h"
#include "SingleLayerNetwork.h"

#include "spaint.h"
#include "Color.h"

using namespace spaint;

#define KEY_ESCAPE 9
#define KEY_R      27

// Function properties
#define FUNCTION_START  0.0
#define FUNCTION_END    1.0
#define FUNCTION_POINTS 100

// Network properties
#define SL_NETWORK_NAME "output/approx.neetwook"
#define NETWORK_ENABLE_OFFSETS     1
#define NETWORK_ACTIVATOR_FUNCTION Linear

// Render approx network result plot

// bash c.sh "-lX11" src/approx_render


class scene : public component {
	
	void create() {
		get_paint().init_font();
		get_window().set_title("Approx plot");
		
		updated = 1;
	};
	
	void destroy() {};
	
	
	// Point set for testring
	std::vector<double> get_plot_set() {
		std::vector<double> points(FUNCTION_POINTS);
		
		double step = (FUNCTION_END - FUNCTION_START) / static_cast<double>(FUNCTION_POINTS + 2);
		int i = 0;
		for (double d = FUNCTION_START + step; d < FUNCTION_END - step; d += step)
			points[i++] = d;
		
		return points;
	};
	
	
	// Tested function definition
	constexpr double get_function(double t) {
		return -1.0 + 2.0 * t; // sin(t * 3.141528); // std::sin(2.0 * 3.141528 * t) * std::cos(5.0 * 3.141528 * t);
	};


	
	bool reload() {
		network.setEnableOffsets(NETWORK_ENABLE_OFFSETS);
		network.setFunction(new NNSpace::NETWORK_ACTIVATOR_FUNCTION());
		
		std::cout << "Deserializing network from " << SL_NETWORK_NAME << std::endl;
		std::ifstream ifs;
		ifs.open(SL_NETWORK_NAME);
		if (ifs.fail()) {
			std::cout << "File " << SL_NETWORK_NAME << " not found" << std::endl;
			return 0;
		}
		if (!network.deserialize(ifs)) {
			std::cout << "Deserialize failed" << std::endl;
			ifs.close();
			return 0;
		}
		ifs.close();
		
		return 1;
	};
	
	NNSpace::SLNetwork network;
	bool mouse_down = 0;
	bool resized = 0;
	bool updated = 0;
	
	void resize() {
		resized = 1;
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
			
			if (updated)
				if (!reload())
					exit(0);
			
			resized = 0;
			updated = 0;
			
			std::vector<double> set = get_plot_set();
			
			p.clear();
			
			for (int i = 0; i < FUNCTION_POINTS; ++i) {
				std::vector<double> input = { set[i] };
				std::vector<double> output = network.run(input);
				
				int x = get_window().get_width() * (set[i]/ (FUNCTION_END - FUNCTION_START));
				int y = get_window().get_height() * ((output[0] + 1.0) * 0.5);
				
				p.color(255, 0, 0);
				p.point(x, y);
				
				p.color(0, 255, 0);
				y = get_window().get_height() * ((get_function(set[i]) + 1.0) * 0.5);
				p.point(x, y);
			}
			
		} else {
			// ...
		}
	};
};


int main() {
	scene s;
	window w(&s, 400, 200, 0);
	w.start();
	return 0;
};

