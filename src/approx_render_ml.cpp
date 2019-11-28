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
#include "MultiLayerNetwork.h"

#include "spaint.h"
#include "Color.h"

using namespace spaint;

#define KEY_ESCAPE 9
#define KEY_R      27

// Function properties
#define DISPLAY_OFFSET          0.25
#define DISPLAY_OFFSET_VERTICAL 0.1

// Render approx network result plot

// bash c.sh "-lX11" src/approx_render_ml
// ./bin/src/approx_render_mc 0.0 1.0 output/ml_network.neetwook

static std::string filename;
static double start_v, end_v;

class scene : public component {
	
	void create() {
		get_paint().init_font();
		get_window().set_title("Approx plot");
		
		updated = 1;
	};
	
	void destroy() {};
	
	// Point set for testring
	std::vector<double> get_plot_set() {
		std::vector<double> points(get_window().get_width());
		
		double interval = (end_v - start_v) / (1.0 - 2.0 * DISPLAY_OFFSET);
		double step = interval / static_cast<double>(get_window().get_width() + 2);
		int i = 0;
		for (double d = start_v - interval * DISPLAY_OFFSET + step; d < end_v + interval * DISPLAY_OFFSET - step; d += step)
			points[i++] = d;
		
		return points;
	};
	
	bool reload() {		
		// std::cout << "Deserializing network from " << filename << std::endl;
		std::ifstream ifs;
		ifs.open(filename);
		if (ifs.fail()) {
			std::cout << "File " << filename << " not found" << std::endl;
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
	
	NNSpace::MLNetwork network;
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
			
			double interval = (end_v - start_v) / (1.0 - 2.0 * DISPLAY_OFFSET);
			double vinterval = 1.0 / (1.0 - 2.0 * DISPLAY_OFFSET_VERTICAL);
			
			p.color(0, 0, 255);
			p.line(get_window().get_width() * DISPLAY_OFFSET, 0, get_window().get_width() * DISPLAY_OFFSET, get_window().get_height());
			p.line(get_window().get_width() * (1.0 - DISPLAY_OFFSET), 0, get_window().get_width() * (1.0 - DISPLAY_OFFSET), get_window().get_height());
			p.line(0, get_window().get_height() / 2, get_window().get_width(), get_window().get_height() / 2);
			p.line(0, get_window().get_height() * DISPLAY_OFFSET_VERTICAL, get_window().get_width(), get_window().get_height() * DISPLAY_OFFSET_VERTICAL);
			p.line(0, get_window().get_height() * (1.0 - DISPLAY_OFFSET_VERTICAL), get_window().get_width(), get_window().get_height() * (1.0 - DISPLAY_OFFSET_VERTICAL));
			
			for (int i = 0; i < set.size(); ++i) {
				std::vector<double> input = { set[i] };
				std::vector<double> output = network.run(input);
				
				int x = get_window().get_width() * ((set[i] - set[0]) / interval);
				int y = get_window().get_height() - get_window().get_height() * ((output[0] + 1.0) * (0.5 - DISPLAY_OFFSET_VERTICAL) + DISPLAY_OFFSET_VERTICAL);
				
				p.color(255, 0, 0);
				p.point(x, y);
			}
			
		} else {
			// ...
		}
	};
};

int main(int argc, char** argv) {
	
	if (argc < 4) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	start_v = std::stod(argv[1]);
	end_v = std::stod(argv[2]);
	filename = argv[3];
	
	scene s;
	window w(&s, 400, 200, 0);
	w.start();
	return 0;
};

