#include<cmath>
#include<cstdint>

struct vec3 {
	float x, y, z;

	vec3() : x{0.0}, y{0.0}, z{0.0} {}
	vec3(float _x, float _y, float _z ) : x{_x}, y{_y}, z{_z} {}
	
	inline vec3 operator-() const { return vec3(-x, -y, -z); }
	inline vec3& operator+=(const vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
	inline vec3& operator-=(const vec3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
	inline vec3& operator*=(const vec3& o) { x*=o.x; y*=o.y; z*=o.z; return *this; }
	inline vec3& operator*=(float o) { x*=o; y*=o; z*=o; return *this; }

	inline float dot(const vec3& o) const { return x*o.x+y*o.y+z*o.z; }
	inline vec3 cross(const vec3& o) const { return vec3(y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x); }
	
	inline float length() const { return sqrtf(dot(*this)); }
	inline float length2() const { return dot(*this); }
	inline vec3 normalized() const { float l = length(); return vec3(x/l, y/l, z/l);/**this * (1.0f / length());*/ }
};

inline vec3 operator+(vec3 l, const vec3& r) { return l+=r; }
inline vec3 operator-(vec3 l, const vec3& r) { return l-=r; }
inline vec3 operator*(vec3 l, const vec3& r) { return l*=r; }
inline vec3 operator*(vec3 l, float r) { return l*=r; }
inline vec3 operator*(float l, vec3 r) { return r*=l; }

inline float dot(const vec3& l, const vec3& r) { return l.dot(r); }
inline vec3 cross(const vec3& l, const vec3& r) { return l.cross(r); }

inline vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2*v.dot(n)*n;
}
inline bool refract(const vec3& v, const vec3& n, float ni, vec3& refracted) {
	float dt = v.dot(n);
	float d = 1.0f - ni*ni*(1.0f - dt*dt);
	if (d > 0) {
		refracted = ni * (v - dt*n) - sqrtf(d)*n;
		return true;
	}
	return false;
}
float schlick(float cosine, float ri) {
	float r0 = (1.0f-ri)/(1.0f+ri);
	r0 = r0*r0;
	return r0 + (1-r0)*powf(1-cosine, 5);
}

struct Sphere {
	vec3 center;
	float radius;
	
	Sphere(vec3 _center, float _radius) : center{_center}, radius{_radius} {}
};

struct Ray {
	vec3 origin;
	vec3 direction;
	
	Ray() {}
	Ray(vec3 _origin, vec3 _direction) : origin{_origin}, direction{_direction} {}
	vec3 pointAt(float l) const { return origin + l * direction; }
};

struct Hit {
	vec3 position;
	vec3 normal;
	float t;
};

#define PI 3.1415926f

namespace random {
	uint32_t xorShift(uint32_t& state) {
		uint32_t x = state;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 15;
		return state = x;
	}
	
	float randFloat(uint32_t& state) {
		return (xorShift(state) & 0xffffff) / 16777216.0f;
	}
	
	vec3 vecInUnitCircle(uint32_t& state) {
		vec3 v;
		while (v.length2() >= 1.0f) {
			v = 2.0f * vec3(randFloat(state), randFloat(state), 0.0f) - vec3(1.0f,1.0f,0.0f);
		}
		return v;
	}
	
	vec3 vecInUnitSphere(uint32_t& state){
		vec3 v;
		while (v.length2() >= 1.0f) {
			v = 2.0 * vec3(randFloat(state), randFloat(state), randFloat(state)) - vec3(1.0f,1.0f,1.0f);
		}
		return v;
	}
	
	vec3 unitVec(uint32_t& state) {
		float z = randFloat(state) * 2.0f - 1.0f;
		float a = randFloat(state) * 2.0f * PI;
		float r = sqrtf(1.0f - z*z);
		float x = r*cosf(a);
		float y = r*sinf(a);
		return vec3(x, y, z);
	}
}

struct Camera {
	vec3 origin;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	vec3 lowerLeftCorner;
	float lensRadius;
	
	Camera() {}
	
	Camera(const vec3& from, const vec3& to, const vec3& up, float fov, float aspect, float aperture, float focusDist){
		lensRadius = aperture / 2.0f;
		float theta = fov * PI / 180.0f;
		float halfHeight = tanf(theta / 2.0f);
		float halfWidth = aspect * halfHeight;
		origin = from;
		w = (from - to).normalized();
		u = up.cross(w).normalized();
		v = w.cross(u);
		lowerLeftCorner = origin - (halfWidth*u + halfHeight*v + w) * focusDist;
		horizontal = 2*halfWidth*u*focusDist;
		vertical = 2*halfHeight*v*focusDist;
	}
	
	Ray getRay(float x, float y, uint32_t& state) const {
		vec3 r = lensRadius * random::vecInUnitCircle(state);
		vec3 offset = u * r.x + v * r.y;
		return Ray(origin+offset, (lowerLeftCorner+x*horizontal+y*vertical-origin-offset).normalized());
	}
};

struct Material {
	enum Type { Lambert, Metal, Dielectric };
	
	Type type;
	vec3 albedo;
	vec3 emissive;
	float rough;
	float ri;
};

const float tmin = 0.001f;
const float tmax = 1.0e7f;

// replicate Aras P scene
const Sphere spheres[] = {
	{ vec3(0,-100.5,-1), 100.0f },
	{ vec3(2,0,-1), 0.5f },
	{ vec3(0,0,-1), 0.5f },
	{ vec3(-2,0,-1), 0.5f },
	{ vec3(2,0,1), 0.5f },
	{ vec3(0,0,1), 0.5f },
	{ vec3(-2,0,1), 0.5f },
	{ vec3(0.5f,1,0.5f), 0.5f },
	{ vec3(-1.5f,1.5f,0), 0.3f }
};
const int spheresCount = sizeof(spheres) / sizeof(spheres[0]);

const Material materials[spheresCount] = {
	{ Material::Lambert, vec3(0.8f, 0.8f, 0.8f), vec3(), 0, 0 },
	{ Material::Lambert, vec3(0.8f, 0.4f, 0.4f), vec3(), 0, 0 },
	{ Material::Lambert, vec3(0.4f, 0.8f, 0.4f), vec3(), 0, 0 },
	{ Material::Metal, vec3(0.4f, 0.4f, 0.8f), vec3(), 0, 0 },
	{ Material::Metal, vec3(0.4f, 0.8f, 0.4f), vec3(), 0, 0 },
	{ Material::Metal, vec3(0.4f, 0.8f, 0.4f), vec3(), 0.2f, 0 },
	{ Material::Metal, vec3(0.4f, 0.8f, 0.4f), vec3(), 0.6f, 0 },
	{ Material::Dielectric, vec3(0.4f, 0.4f, 0.4f), vec3(), 0, 1.5f },
	{ Material::Lambert, vec3(0.8f, 0.6f, 0.2f), vec3(30,25,15), 0, 0 }
};
int emitersIds[spheresCount];
int emitersCount = 0;


int hitWorld(const Ray& ray, Hit& hit, int& id) {
	float hitT = tmax;
	id = -1;
	for (int i = 0; i < spheresCount; ++i){
		vec3 co = spheres[i].center - ray.origin;
		float nb = co.dot(ray.direction);
		float c = co.length2() - spheres[i].radius*spheres[i].radius;
		float d = nb*nb - c;
		if (d > 0) {
			d = sqrtf(d);
			
			float t = nb - d;
			if (t <= tmin)
				t = nb + d;
			if (t > tmin && t < hitT) {
				id = i;
				hitT = t;
			}
		}
	}
	if (id != -1) {
		hit.position = ray.pointAt(hitT);
		hit.normal = (hit.position - spheres[id].center)*(1.0f/spheres[id].radius);
		hit.t = hitT;
	}
	return id;
}

#define MAX_RAY_DEPTH 10
#define ENV_LIGHT vec3(0.15f,0.21f,0.3f)

bool scatter(const Ray& ray, const Material& material, const Hit& hit, Ray& scattered, vec3& attenuation, vec3& lightEmission, int& rayCount, uint32_t& state) {
	attenuation = material.albedo;
	lightEmission = vec3();
	
	switch(material.type){
	case Material::Lambert:
	{
		vec3 target = /*hit.position +*/ hit.normal + random::unitVec(state);
		scattered = Ray(hit.position, (target /*- hit.position*/).normalized());
		
		for (int i = 0; i < emitersCount; i++){
			int id = emitersIds[i];
			const Material& emat = materials[id];
			if (&material == &emat) continue;
			const Sphere& s = spheres[id];
			
			vec3 sw = (s.center - hit.position).normalized();
			vec3 su = (fabs(sw.x)>0.01f ? vec3(0.0,1.0,0.0) : vec3(1.0,0.0,0.0)).cross(sw).normalized();
			vec3 sv = sw.cross(su);
			
			float cosMax = sqrtf(1.0f - s.radius*s.radius / (hit.position - s.center).length2());
			float eps1 = random::randFloat(state), eps2 = random::randFloat(state);
			float cosA = 1.0f - eps1 + eps1 * cosMax;
			float sinA = sqrtf(1.0f - cosA*cosA);
			float phi = 2 * PI * eps2;
			vec3 l = su * (cosf(phi) * sinA) + sv * (sinf(phi) * sinA) + sw * cosA;
			
			Hit lightHit;
			int hitId;
			rayCount++;
			if (hitWorld(Ray(hit.position, l), lightHit, hitId) != -1 && hitId == id) {
				float omega = 2 * PI * (1 - cosMax);
				vec3 dir = ray.direction;
				vec3 nl = hit.normal.dot(dir) < 0 ? hit.normal : - hit.normal;
				lightEmission += (material.albedo * emat.emissive) * (fmax(0.0f, l.dot(nl)) * omega / PI);
			}
		}
		
		return true;
	}
		break;
	case Material::Metal:
	{
		vec3 reflection = reflect(ray.direction, hit.normal);
		scattered = Ray(hit.position, (reflection + material.rough*random::vecInUnitSphere(state)).normalized());
		return scattered.direction.dot(hit.normal) > 0;
	}
		break;
	case Material::Dielectric: 
	{
		vec3 dir = ray.direction;
		vec3 reflection = reflect(dir, hit.normal);
		attenuation = vec3(1.0f,1.0f,1.0f);
		
		vec3 outN = hit.normal;
		float ni = material.ri;
		float cosine = dir.dot(hit.normal);
		if(dir.dot(hit.normal) > 0) {
			outN = -outN;
			cosine *= material.ri;
		} else {
			ni = 1.0f / ni;
			cosine = -cosine;
		}
		
		vec3 refraction;
		float reflectionP;
		if(refract(dir, outN, ni, refraction)) {
			reflectionP = schlick(cosine, material.ri);
		} else {
			reflectionP = 1.0f;
		}
		
		if(random::randFloat(state) < reflectionP) {
			scattered = Ray(hit.position, reflection);
		} else {
			scattered = Ray(hit.position, refraction);
		}
		return true;
	}
		break;
	default:
		attenuation = vec3(1.0f,0.0f,1.0f);
		return false;
	}
	return true;
}

vec3 trace(const Ray& ray, int depth, int& rayCount, uint32_t& state, bool isEmissiveMaterial = true) {
	Hit hit;
	int id = 0;
	
	rayCount++;
	if (hitWorld(ray, hit, id) != -1) {
		Ray scattered;
		vec3 attenuation, lightEmission;
		const Material& material = materials[id];
		vec3 materialEmission = material.emissive;
		
		if (depth < MAX_RAY_DEPTH && scatter(ray, material, hit, scattered, attenuation, lightEmission, rayCount, state)) {
			if (!isEmissiveMaterial) materialEmission = vec3();
			isEmissiveMaterial = (material.type != Material::Lambert);
			
			return materialEmission + lightEmission + attenuation * trace(scattered, depth+1, rayCount, state, isEmissiveMaterial);
		} else {
			return materialEmission;
		}
	} else {
		return ENV_LIGHT;
	}
}


struct Scene {
	int screenWidth, screenHeight;
	int samplesPerPixel;
	int frameCount = 0;
	int rayCount = 0;
	Camera camera;
	float* imgBuf;
};

void raytrace(Scene& scene) {
	float* p = scene.imgBuf;
	#pragma omp parallel for shared(scene,p)
	for (int y = 0; y < scene.screenHeight; ++y) {
		uint32_t state = (y * 9781 + scene.frameCount * 6271) | 1;
		for (int x = 0; x < scene.screenWidth; ++x) {
			vec3 color;
			for (int i = 0; i < scene.samplesPerPixel; ++i) {
				float x_of = float(x + random::randFloat(state)) / scene.screenWidth;
				float y_of = float(y + random::randFloat(state)) / scene.screenHeight;
				
				Ray r = scene.camera.getRay(x_of, y_of, state);
				color += trace(r, 0, scene.rayCount, state);
			}
			color *= 1.0f / float(scene.samplesPerPixel);
			// write color
			int i = (y * scene.screenWidth + x) * 4; 
			p[i+0] = color.x;
			p[i+1] = color.y;
			p[i+2] = color.z;
			//p += 4;
		}
	}
}

#include<fstream>
namespace {
	Scene& setupScene(Scene& scene) {
		scene.screenHeight = 720;
		scene.screenWidth = 1280;
		scene.samplesPerPixel = 128;
		scene.imgBuf = new float[scene.screenHeight*scene.screenWidth*4];
		
		for (int i = 0; i < spheresCount; ++i) {
			const Material& material = materials[i];
			if (material.emissive.x > 0 || material.emissive.y > 0 || material.emissive.z > 0) {
				emitersIds[emitersCount++] = i;
			}
		}
		
		scene.camera = Camera(vec3(0,2,3), vec3(), vec3(0,1,0), 60, scene.screenWidth/float(scene.screenHeight), 0.1f, 3);
		return scene;
	}
	
	#define MIN(a,b) ((a)<(b)?(a):(b))
	
	unsigned int linear2srgb(float x) {
		x = fmax(x, 0.0f);
		x = fmax(1.055f * float(powf(x, 0.4166666667f)) - 0.055f, 0.0f);
		return MIN((unsigned int)(x * 255.9f), 255u);
	}
	
	void saveImage(const Scene& scene) {
		float* p = scene.imgBuf;
		
		int bufLen = scene.screenWidth*scene.screenHeight*4;
		unsigned char* img = new unsigned char[bufLen];
		for (int i =0; i < bufLen; i+=4) {
			img[i] = (unsigned char)linear2srgb(p[i + 2]);
			img[i + 1] = (unsigned char)linear2srgb(p[i + 1]);
			img[i + 2] = (unsigned char)linear2srgb(p[i]);
			img[i + 3] = 255;
		}
		
		unsigned char header[] =  {
			0,0,2,0,0,0,0,0,0,0,0,0,
			(unsigned char)(scene.screenWidth & 0x00ff),
			(unsigned char)((scene.screenWidth & 0xff00) >> 8),
			(unsigned char)(scene.screenHeight & 0x00ff),
			(unsigned char)((scene.screenHeight & 0xff00) >> 8),
			32, 0
		};
		
		std::ofstream imgFile("result.tga", std::ios::binary);
		imgFile.write((char*)header, sizeof(header));
		imgFile.write((char*)img, bufLen);
		
		delete[] img;
	}
}

#include<iostream>
#include<chrono>

int main() {
	Scene scene;
	setupScene(scene);
	
	std::cout << "Scene generated: " << spheresCount 
			<< " object(s), " << emitersCount << " emiter(s)." << std::endl
			<< "Rendering..." << std::endl;
	auto start_t = std::chrono::steady_clock::now();
	
	raytrace(scene);
	
	auto duration = std::chrono::steady_clock::now() - start_t;
	auto h = std::chrono::duration_cast<std::chrono::hours>(duration);
	auto m = std::chrono::duration_cast<std::chrono::minutes>(duration -= h);
	auto s = std::chrono::duration_cast<std::chrono::seconds>(duration -= m);
	std::cout << "Rendering complete in ";
	if (h.count() > 0) std::cout << h.count() << " h ";
	if (m.count() > 0) std::cout << m.count() << " m ";
	std::cout << s.count() << " s" << std::endl;
	
	float MRays = scene.rayCount / 1.0e6f;
	float MRaysPerSec = MRays / (s+m+h).count();
	std::cout << MRays << " Mrays, " << MRaysPerSec << " Mray/sec" << std::endl;
	
	saveImage(scene);
	delete[] scene.imgBuf;
	
	std::cin.ignore(255, '\n');	
	return 0;
}
