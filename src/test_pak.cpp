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

#include <cmath>
#include <unistd.h>
#include <time.h>

#include "spaint.h"
#include "Color.h"
#include "mnist/mnist_reader.hpp"

using namespace spaint;

#define KEY_ESCAPE 9
#define KEY_R      27
#define MNIST_DATA_LOCATION "input"
#define UPDATE_STEP 100

// Preview dataset testing images with labels

// bash c.sh "-lX11" src/test_pak


class scene : public component {
	
	void create() {
		get_paint().init_font();
		get_window().set_title("Unpack example");
		
		dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
		
		updated = 1;
		cloookk = clock();
	};
	
	void destroy() {
		
	};
	
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
	bool mouse_down = 0;
	bool resized = 0;
	bool updated = 0;
	int image_id = 0;
	
	long step = 100;
	clock_t cloookk;
	
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
			
		if (resized || updated || mouse_down) {
			
			resized = 0;
			updated = 0;
			
			if (resized || updated || clock() - cloookk > UPDATE_STEP) {
				p.clear();
				
				p.color(Color::WHITE);
				
				std::string lab = std::to_string(dataset.training_labels[image_id]);
				p.text(30, 10, lab.c_str());
				
				for (int x = 0; x < 28; ++x)
					for (int y = 0; y < 28; ++y) 
						if (dataset.training_images[image_id][x + y * 28] == 0)
							p.point(x, y);
				
				++image_id;
				image_id = image_id % dataset.training_images.size();
				cloookk = clock();
			}
		} else {
			// ...
		}
	};
};


int main() {
	scene s;
	window w(&s, 40, 40, 0);
	w.start();
	return 0;
};

