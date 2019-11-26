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

#include <random>
#include <fstream>
#include <cmath>
#include <unistd.h>
#include <algorithm>

#include "Network.h"
#include "MultiLayerNetwork.h"

#include "spaint.h"
#include "Color.h"

using namespace spaint;

#define KEY_ESCAPE 9
#define KEY_R      27

// Tweaks
#define MNIST_DATA_LOCATION "input"
#define NETWORK_SERIALIZE_PATH "output/approx.neetwook"

// Training & testing properties
#define START_TRAIN_RATE  0.01
#define SCALE_TRAIN_RATE  1.0
#define TRAIN_SET_SIZE    100000
#define TEST_SET_SIZE     100
#define ACCURATE_SET_SIZE 100
#define TRAIN_ALGO_ID     0
#define INIT_ALGO_ID      0

#define FUNCTION_START    0.0
#define FUNCTION_END      1.0
#define FUNCTION_FUNC     std::sin(2.0 * 3.141528 * t) * 0.5 + 0.5

#define NETWORK_ACTIVATOR_FUNCTION TanH
#define NETWORK_TOPO               { 1, 3, 1 }
#define NETWORK_ENABLE_OFFSETS     1
#define TEST_PASS_GRADE            0.01

#define DISPLAY_OFFSET          0.25
#define DISPLAY_OFFSET_VERTICAL 0.1

#define REPAINT_STEP       1000
#define REPAINT_STEP_DELAY 0

#define RANDOMIZE_SET

// Network properties
#define SL_NETWORK_NAME "output/approx.neetwook"
#define NETWORK_ENABLE_OFFSETS 1

// Render approx network result plot

// bash c.sh "-lX11" src/approx_realtime_multilayer_render


class scene : public component {
	
	void create() {
		get_paint().init_font();
		get_window().set_title("Approx multiline plot");
		
		updated = 1;
		
		network.initialize(INIT_ALGO_ID);
		network.setEnableOffsets(NETWORK_ENABLE_OFFSETS);
		network.setActivator(new NNSpace::NETWORK_ACTIVATOR_FUNCTION());
		
		train_set = get_train_set();
		
#ifdef RANDOMIZE_SET
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(train_set.begin(), train_set.end(), g);
#endif
	};
	
	void destroy() {};
	
	// Point set for training
	std::vector<double> get_train_set() {
		std::vector<double> points(TRAIN_SET_SIZE);
		
		double step = (FUNCTION_END - FUNCTION_START) / static_cast<double>(TRAIN_SET_SIZE + 1);
		for (int i = 0; i < TRAIN_SET_SIZE; ++i)
			points[i] = ((double) i) * step;
		
		return points;
	};

	// Point set for testring
	std::vector<double> get_test_set() {
		std::vector<double> points(TEST_SET_SIZE);
		
		double step = (FUNCTION_END - FUNCTION_START) / static_cast<double>(TEST_SET_SIZE + 1);
		for (int i = 0; i < TEST_SET_SIZE; ++i)
			points[i] = ((double) i) * step + step * 0.3;
		
		return points;
	};

	// Point set for accurete testing, differs from test set, smaller
	std::vector<double> get_accurate_set() {
		std::vector<double> points(ACCURATE_SET_SIZE);
		
		double step = (FUNCTION_END - FUNCTION_START) / static_cast<double>(ACCURATE_SET_SIZE + 1);
		for (int i = 0; i < ACCURATE_SET_SIZE; ++i)
			points[i] = ((double) i) * step + step * 0.6;
		
		return points;
	};

	// Tested function definition
	constexpr double get_function(double t) {
		return FUNCTION_FUNC; // -1.0 + 2.0 * t; // sin(t * 3.141528); // std::sin(2.0 * 3.141528 * t) * std::cos(5.0 * 3.141528 * t);
	};

	
	// Point set for testring
	std::vector<double> get_plot_set() {
		std::vector<double> points(get_window().get_width());
		
		double interval = (FUNCTION_END - FUNCTION_START) / (1.0 - 2.0 * DISPLAY_OFFSET);
		double step = interval / static_cast<double>(get_window().get_width() + 2);
		int i = 0;
		for (double d = FUNCTION_START - interval * DISPLAY_OFFSET + step; d < FUNCTION_END + interval * DISPLAY_OFFSET - step; d += step)
			points[i++] = d;
		
		return points;
	};
	
	NNSpace::MLNetwork network = NNSpace::MLNetwork(NETWORK_TOPO);
	bool mouse_down = 0;
	bool resized    = 0;
	bool updated    = 0;
	bool painting   = 1;
	double rate     = START_TRAIN_RATE;
	int step        = 0;
	
	std::vector<double> train_set;
	
	void resize() {
		resized = 1;
	};
	
	void loop() {
		window& w = get_window();
		painter& p = w.get_paint();
		
		// Block untill event is reached
		// if (!mouse_down) w.wait_event(1);
		
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
		
		// Make train step
		if (step < train_set.size()) {
			std::cout << "Train " << step << " / " << TRAIN_SET_SIZE << std::endl;
			std::vector<double> input = { train_set[step] };
			std::vector<double> output = { get_function(train_set[step]) };
			
			rate = std::fabs(network.train_error(TRAIN_ALGO_ID, input, output, rate)) * SCALE_TRAIN_RATE;
			rate = rate <= START_TRAIN_RATE ? START_TRAIN_RATE : rate;
			
			++step;
		}
		
		// Render step
		if (resized || updated || painting && (step % REPAINT_STEP == 0 || step >= train_set.size() - 1)) {
			
			if (step >= train_set.size() - 1)
				painting = 0;
			
			if (updated) {
				painting = 1;
				network.initialize(INIT_ALGO_ID);
				
#ifdef RANDOMIZE_SET
				std::random_device rd;
				std::mt19937 g(rd());
				std::shuffle(train_set.begin(), train_set.end(), g);
#endif

				resized = 0;
				updated = 0;
				
				step = 0;
				
				return;
			}
			
			resized = 0;
			updated = 0;
			
			std::vector<double> set = get_plot_set();
			
			p.clear();
			
			double interval = (FUNCTION_END - FUNCTION_START) / (1.0 - 2.0 * DISPLAY_OFFSET);
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
				
				p.color(0, 255, 0);
				y = get_window().get_height() - get_window().get_height() * ((get_function(set[i]) + 1.0) * (0.5 - DISPLAY_OFFSET_VERTICAL) + DISPLAY_OFFSET_VERTICAL);
				p.point(x, y);
			}
			
			usleep(REPAINT_STEP_DELAY * 1000);
			
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

