#pragma once

namespace spaint {
	// Class used for hording rgb color values
	class Color {
		
	public:
		
		int r = 0, g = 0, b = 0, a = 255;
		
		static Color RED;
		static Color YELLOW;
		static Color GREEN;
		static Color CYAN;
		static Color BLUE;	
		static Color MAGENTA;	
		static Color BLACK;
		static Color WHITE;
		
		Color() {};
		
		Color(int v) {
			r = v;
			g = v;
			b = v;
		};
		
		Color(int v, int opacity) {
			r = v;
			g = v;
			b = v;
			a = opacity;
		};
		
		Color(int r, int g, int b) {
			this->r = r;
			this->g = g;
			this->b = b;
		};
		
		Color(int r, int g, int b, int a) {
			this->r = r;
			this->g = g;
			this->b = b;
			this->a = a;
		};
		
		static Color fromRGB(int rgb) {
			Color c;
			c.r = (rgb >> 16) & 0xFF;
			c.g = (rgb >>  8) & 0xFF;
			c.b = (rgb >>  0) & 0xFF;
			return c;
		};
		
		static Color fromARGB(int argb) {
			Color c;
			c.a = (argb >> 24) & 0xFF;
			c.r = (argb >> 16) & 0xFF;
			c.g = (argb >>  8) & 0xFF;
			c.b = (argb >>  0) & 0xFF;
			return c;
		};
		
		int rgb() {
			return (r << 16) | (g << 8) | (b << 0);
		};
		
		int argb() {
			return (a << 24) | (r << 16) | (g << 8) | (b << 0);
		};
		
		void normalize() {
			r = r < 0 ? 0 : (r > 255 ? 255 : r);
			g = g < 0 ? 0 : (g > 255 ? 255 : g);
			b = b < 0 ? 0 : (b > 255 ? 255 : b);
			a = a < 0 ? 0 : (a > 255 ? 255 : a);
		};
		
		// Interpolate between two colors
		static Color interpolate(const Color& a, const Color& b, double t) {
			Color c;
			c.r = a.r * (1 - t) + b.r * t;
			c.g = a.g * (1 - t) + b.g * t;
			c.b = a.b * (1 - t) + b.b * t;
			c.a = a.a * (1 - t) + b.a * t;
			return c;
		};
		
		static Color halfmix(const Color& a, const Color& b) {
			Color c;
			c.r = (a.r + b.r) >> 1;
			c.g = (a.b + b.g) >> 1;
			c.b = (a.b + b.b) >> 1;
			c.a = (a.a + b.a) >> 1;
			return c;
		};
		
		static Color halfmix3(const Color& a, const Color& b, const Color& c) {
			Color d;
			d.r = (a.r + b.r + c.r) / 3;
			d.g = (a.b + b.g + c.g) / 3;
			d.b = (a.b + b.b + c.b) / 3;
			d.a = (a.a + b.a + c.a) / 3;
			return d;
		};
		
		static Color halfmix4(const Color& a, const Color& b, const Color& c, const Color& d) {
			Color e;
			e.r = (a.r + b.r + c.r + d.r) >> 2;
			e.g = (a.b + b.g + c.g + d.g) >> 2;
			e.b = (a.b + b.b + c.b + d.b) >> 2;
			e.a = (a.a + b.a + c.a + d.a) >> 2;
			return e;
		};
		
		int& operator[](const int chrome) {
			int chrome_ = chrome & 0b11;
			if (chrome_ == 0)
				return r;
			if (chrome_ == 1)
				return g;
			if (chrome_ == 2)
				return b;
			if (chrome_ == 3)
				return a;
		};
	};
		
	Color operator+(const Color& a, const Color& b) {
		Color c(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a);
		c.normalize();
		return c;
	};
	
	Color Color::RED     = Color(255, 0  , 0  );
	Color Color::YELLOW  = Color(255, 255, 0  );
	Color Color::GREEN   = Color(0  , 255, 0  );
	Color Color::CYAN    = Color(0  , 255, 255);
	Color Color::BLUE    = Color(0  , 0  , 255);
	Color Color::MAGENTA = Color(255, 0  , 255);
	Color Color::BLACK   = Color(0  , 0  , 0  );
	Color Color::WHITE   = Color(255, 255, 255);	
};