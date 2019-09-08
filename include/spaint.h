/*
    primitive drawing framework using xlib
	
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

#pragma once



#include <iostream>




#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdlib.h>
#include <functional>
#include <cstring>
#include <exception>

#include "Color.h"

// Require $(pkg-config --cflags --libs x11) linking

// Simple paint
namespace spaint {
	
	// Used as wrapper for XPoint
	struct Point {
		int x = 0;
		int y = 0;
		
		Point(int x, int y) {
			this->x = x;
			this->y = y;
		};
		
		Point(int r) {
			x = y = r;
		};
		
		Point() {
			x = y = 0;
		};
		
		operator XPoint() const {
			XPoint p;
			p.x = x;
			p.y = y;
			return p;
		};
	};
	
	// Scene wrapper
	class window;
	class painter;
	class component {
		friend window;
		
	
	public:
		window* win;
	
		inline window& get_window() {
			return *win;
		};
	
		inline painter& get_paint();
	
		virtual void start() {};
		virtual void stop() {};
		
		virtual void create() {};
		virtual void destroy() {};
		
		virtual void resize() {};
		
		virtual void loop() {};
	};
	
	class ImageBuffer {
		friend class painter;
		
		Display *display;
		Window       win;
		GC            gc;
		XImage*    image;
		int width, height;
		char*        data;
		XColor    current;
		
	public:
		
		ImageBuffer() {};
	
		inline int get_width() {
			return width;
		};
	
		inline int get_height() {
			return height;
		};
	
		// Draw current data to display
		inline void put(int x, int y, int offset_x = 0, int offset_y = 0, int sub_x = 0, int sub_y = 0) {
			XPutImage(display, win, gc, image, offset_x, offset_y, x, y, width - sub_x, height - sub_y);
		};
		
		int set_color(const Color& color) {
			current.flags = DoRed | DoGreen | DoBlue;
			current.red   = color.r << 8;
			current.green = color.g << 8;
			current.blue  = color.b << 8;
			Status rc = XAllocColor(display, DefaultColormap(display, DefaultScreen(display)), &current);
			if (rc == 0)
				return 0;
			return 1;
		};
		
		void fill() {
			for (int x = 0; x < width; ++x)
				for (int y = 0; y < height; ++y)
					XPutPixel(image, x, y, current.pixel);
		};
		
		void set_pixel_unsafe(int x, int y) {
			XPutPixel(image, x, y, current.pixel);
		};
		
		void set_pixel(int x, int y) {
			if (x < 0 || x >= width || y < 0 || y >= height)
				return;
			
			XPutPixel(image, x, y, current.pixel);
		};
		
		Color get_pixel_unsafe(int x, int y) {			
			Color result;
			XColor color;
			
			color.pixel = XGetPixel(image, x, y);
			XQueryColor(display, DefaultColormap(display, DefaultScreen(display)), &color);
			
			result.r = color.red;
			result.g = color.green;
			result.b = color.blue;
			
			return result;
		};
		
		Color get_pixel(int x, int y) {
			if (x < 0 || x >= width || y < 0 || y >= height)
				return Color(0, 0, 0);
			
			Color result;
			XColor color;
			
			color.pixel = XGetPixel(image, x, y);
			XQueryColor(display, DefaultColormap(display, DefaultScreen(display)), &color);
			
			result.r = color.red;
			result.g = color.green;
			result.b = color.blue;
			
			return result;
		};
	};
	
	class painter {
		friend class window;
		
		Display *display;
		Window       win;
		GC            gc;
		XGCValues values;
		XFontStruct* font = nullptr;
		
		Colormap cmap;
		XColor current;
		
	public:
	
		inline bool color(int r, int g, int b) {
			current.flags = DoRed | DoGreen | DoBlue;
			current.red = r << 8;
			current.green = g << 8;
			current.blue = b << 8;
			Status rc = XAllocColor(display, cmap, &current);
			if (rc == 0)
				return 0;
			XSetForeground(display, gc, current.pixel);
			return 1;
		};
		
		inline bool color(const Color& c) {
			current.flags = DoRed | DoGreen | DoBlue;
			current.red = c.r << 8;
			current.green = c.g << 8;
			current.blue = c.b << 8;
			Status rc = XAllocColor(display, cmap, &current);
			if (rc == 0)
				return 0;
			XSetForeground(display, gc, current.pixel);
			return 1;
		};
		
		ImageBuffer createImageBuffer(int width, int height) {
			int def_depth = DefaultDepth(display, DefaultScreen(display));
			int bit_depth = def_depth >> 3;
			
			ImageBuffer ib;
			char* data = (char*) malloc(32 * width * height);
			if (!data)
				throw std::runtime_error("can not allocate image data");
			
			ib.image   = XCreateImage(display, CopyFromParent, def_depth, XYPixmap, 0, data, width, height, 32, 0);
			if (!ib.image)
				throw std::runtime_error("failed creating ImageBuffer");
			
			ib.width   = width;
			ib.height  = height;
			ib.display = display;
			ib.win     = win;
			ib.gc      = gc;
			
			return ib;
		};
		
		void destroyImageBuffer(ImageBuffer& ib) {
			XDestroyImage(ib.image);
		};
		
		inline void clear() {
			XClearWindow(display, win);
		};
		
		inline void clear_rect(int x, int y, int width, int height) {
			XClearArea(display, win, x, y, width, height, 0);
		};
		
		inline void flush() {
			XFlush(display);
		};
		
		inline void point(int x, int y) {
			XDrawPoint(display, win, gc, x, y);
		};
		
		inline void line(int x1, int y1, int x2, int y2) {
			XDrawLine(display, win, gc, x1, y1, x2, y2);
		};
		
		inline void arc(int x, int y, int width, int height, int angle1 = 0, int angle2 = 360 * 64) {
			XDrawArc(display, win, gc, x, y, width, height, angle1, angle2);
		};
		
		inline void fill_rect(int x, int y, int width, int height) {
			XFillRectangle(display, win, gc, x, y, width, height);
		};
		
		inline void line_style(int line_width, int line_style = LineSolid, int cap_style = CapButt, int join_style = JoinBevel) {
			XSetLineAttributes(display, gc, line_width, line_style, cap_style, join_style);
		};
		
		inline void fill_style(int style = FillSolid) {
			XSetFillStyle(display, gc, style);
		};
	
		void init_font(const char* font_name = "fixed") {
			font = XLoadQueryFont(display, font_name);
			XSetFont(display, gc, font->fid);
		};
		
		inline void text(int x, int y, const char *string) {
			if (!font)
				throw std::runtime_error("font not loaded");
			XDrawString(display, win, gc, x, y, string, strlen(string));
		};
		
		// See: https://tronche.com/gui/x/xlib/graphics/filling-areas/XFillPolygon.html
		inline void fill_poly(XPoint *points, int n, int shape = Complex, int mode = CoordModeOrigin) {
			XFillPolygon(display, win, gc, points, n, shape, mode);
		};
		
		inline int text_width(const char *string) {
			return XTextWidth(font, string, strlen(string));
		};
	};
	
	class window {		
		XEvent evt;
		bool has_event = 0;
		
		// Extract next event
		void pump_event(bool wait_for = 0, bool host_events = 0) {	
			has_event = 0;
			while (wait_for || XPending(paint.display)) {
				XNextEvent(paint.display, &evt);
				if (evt.type == ClientMessage) { // Pump quit
					if (evt.xclient.data.l[0] == wmDelete) {
						state = 0;
						return;
					}
				} else if (evt.type == ConfigureNotify) // Pump resize
					if (evt.xconfigure.width != width || evt.xconfigure.height != height) {
						width = evt.xconfigure.width;
						height = evt.xconfigure.height;
						
						if (comp)
							comp->resize();
						
						if (host_events)
							return;
						
						continue;
					}
					
				has_event = 1;
				break;
			}
		};
		
		unsigned int white;
		unsigned int black;
		
		// Reference to component
		component* comp = nullptr;
		
		// Screen number
		int screen;
		
		// Default background
		bool background;
		
		// Delete event
		Atom wmDelete;
		
		// Is running
		bool state = 1;
		
	public:
		
		int width;
		int height;
		
		painter paint;
		
		window() {};
		
		window(component* _comp, int _width, int _height, bool _background = 1) : comp(_comp), 
																					width(_width), 
																					height(_height), 
																					background(_background) {
			if (comp == nullptr)
				throw std::runtime_error("component is null");
							
			comp->win = this;
																						
			paint.display = XOpenDisplay(nullptr);
			if(!paint.display) 
				throw std::runtime_error("failed open display"); 

			screen = DefaultScreen(paint.display);
			white = WhitePixel(paint.display, screen);
			black = BlackPixel(paint.display, screen);


			// Create window
			paint.win = XCreateSimpleWindow(paint.display, DefaultRootWindow(paint.display),
										0, 0, 
										width, height, // size
										0, black,      // border width/clr
										_background ? white : black); // backgrd clr


			// Window close event
			wmDelete = XInternAtom(paint.display, "WM_DELETE_WINDOW", true);
			XSetWMProtocols(paint.display, paint.win, &wmDelete, 1);
			
			
			// Create GC
			unsigned long valuemask = 0;
			
			paint.gc = XCreateGC(paint.display, paint.win, valuemask, &paint.values);
			if (paint.gc < 0)
				throw std::runtime_error("failed create GC");
			
			// Allocate foreground and background colors for this GC
			if (!_background) {
				XSetForeground(paint.display, paint.gc, WhitePixel(paint.display, screen));
				XSetBackground(paint.display, paint.gc, BlackPixel(paint.display, screen));
			} else {
				XSetForeground(paint.display, paint.gc, BlackPixel(paint.display, screen));
				XSetBackground(paint.display, paint.gc, WhitePixel(paint.display, screen));
			}
			
			XSync(paint.display, false);
			
			
			// Create colormap
			paint.cmap = DefaultColormap(paint.display, screen);
			

			// Input
			long eventMask = StructureNotifyMask;
			eventMask |= ButtonPressMask | ButtonReleaseMask // Mouse
			          | KeyPressMask     | KeyReleaseMask;
			XSelectInput(paint.display, paint.win, eventMask);
			
			
			// Show window
			XMapWindow(paint.display, paint.win);
			XFlush(paint.display);
			
			comp->create();
		};
		
		~window() {
			XFreeGC(paint.display, paint.gc);
			XCloseDisplay(paint.display);
		};
		
		window& operator=(const window& w) {
			this->evt = w.evt;
			this->has_event = w.has_event;
			this->white = w.white;
			this->black = w.black;
			this->comp = w.comp;
			this->screen = w.screen;
			this->background = w.background;
			this->wmDelete = w.wmDelete;
			this->state = w.state;
			this->width = w.width;
			this->height = w.height;
			this->paint = w.paint;
			
			// Pass pointer to new window on assignment
			if (this->comp)
				this->comp->win = this;
		};
		
		inline void set_title(char* title) {
			XStoreName(paint.display, paint.win, title);
		};
		
		inline painter& get_paint() {
			return paint;
		};
		
		void start() {
			comp->start();
			
			while (state) {
				check_event();
				
				comp->loop();
			}
			
			comp->stop();
			comp->destroy();
		};
	
		void stop() {
			state = 0;
		};
	
		int get_width() {
			return width;
		};
		
		int get_height() {
			return height;
		};
		
		
		// E V E N T _ H A D N L I N G
		
		
		// Check for new events, skip quit & resize
		inline bool check_event() {
			if (has_event)
				return 1;
			pump_event();
			return has_event;
		};
		
		inline bool next_event() {
			pump_event();
			return has_event;
		};
		
		// Get event
		inline XEvent& get_event() {
			has_event = 0;
			return evt;
		};
		
		inline void clear_events() {
			while (check_event()) get_event();
		};
		
		inline void wait_event(bool host_events = 0) {
			// host_events ~ unblock when host event got reached
			// aka unblock on resize/quit
			if (check_event())
				return;
			
			pump_event(1, host_events);
		};
		
		// Scrolling
		int get_scroll() { 
			if (evt.type == ButtonPress || evt.type == ButtonRelease)
				if (evt.xbutton.button == Button4)
					return 1;
				else if (evt.xbutton.button == Button5)
					return -1;
			return 0;
		};
	
		int get_scroll_x() {
			if (evt.type == ButtonRelease)
				return evt.xbutton.x;
		};
	
		int get_scroll_y() {
			if (evt.type == ButtonRelease)
				return evt.xbutton.y;
		};
	
		bool has_scroll_event(bool ignore_other = 1) { // pop all events till expected reached
			while (check_event()) {
				if ((evt.type == ButtonPress || evt.type == ButtonRelease) && (evt.xbutton.button == Button4 || evt.xbutton.button == Button5)) {
					get_event();
					return 1;
				} else if (!ignore_other) 
					return 0;
				get_event();
			}
			return 0;
		};
	
	
		// Mouse buttons
		bool has_mouse_event(bool ignore_other = 1) {
			while (check_event()) {
				if ((evt.type == ButtonPress || evt.type == ButtonRelease) && !(evt.xbutton.button == Button4 || evt.xbutton.button == Button5)) {
					get_event();
					return 1;
				} else if (!ignore_other)
					return 0;
				get_event();
			}
			return 0;
		};
		
		int get_mouse_x() {
			if (evt.type == ButtonPress || evt.type == ButtonRelease)
				return evt.xbutton.x;
			return -1;
		};
		
		int get_mouse_y() {
			if (evt.type == ButtonPress || evt.type == ButtonRelease)
				return evt.xbutton.y;
			return -1;
		};
	
		// Returns number of pressed button
		int get_button_down() {
			if (evt.type == ButtonPress)
				return evt.xbutton.button;
			return -1;
		};
		
		// Returns number of released button
		int get_button_up() {
			if (evt.type == ButtonRelease)
				return evt.xbutton.button;
			return -1;
		};
	
	
		// Keyboard keys
		bool has_key_event(bool ignore_other = 1) {
			while (check_event()) {
				if (evt.type == KeyPress || evt.type == KeyRelease) {
					get_event();
					return 1;
				} else if (!ignore_other)
					return 0;
				get_event();
			}
			return 0;
		};
		
		// Returns number of pressed key
		int get_key_down() {
			if (evt.type == KeyPress)
				return evt.xkey.keycode;
			return -1;
		};
		
		// Returns number of released key
		int get_key_up() {
			if (evt.type == KeyRelease)
				return evt.xkey.keycode;
			return -1;
		};
	
	
		// Getting mouse & window coordinates
		// Result of get_pointer
		struct pointer {
			// pointer relative location
			int x, y,
				// window location
				win_x, win_y;
			
			pointer(int x, int y, int win_x, int win_y) {
				this->x = x;
				this->y = y;
				this->win_x = win_x;
				this->win_y = win_y;
			};
		};
		
		pointer get_pointer() {
			int win_x, win_y, root_x, root_y = 0;
			unsigned int mask = 0;
			Window child_win, root_win;
			XQueryPointer(paint.display, paint.win,
							&child_win, &root_win,
							&root_x, &root_y, &win_x, &win_y, &mask);
			return pointer(win_x, win_y, root_x, root_y);
		};
	};
	
	painter& component::get_paint() {
		return win->paint;
	};
};
