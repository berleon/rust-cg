extern crate cgmath;
extern crate image;

use std::vec::Vec;
use std::f64;
use std::f64::consts::PI;
use std::path::Path;
use image::{ImageBuffer, Rgb};
use cgmath::{Vector3, InnerSpace, MetricSpace, dot, Zero};


struct PhongMaterial {
    specular: f64,
    diffuse: f64,
    ambient: f64,
    specular_alpha: f64,
}

impl PhongMaterial {
    fn new(specular: f64, diffuse: f64, ambient: f64, specular_alpha: f64) -> PhongMaterial {
        return PhongMaterial {
            specular: specular,
            diffuse: diffuse,
            ambient: ambient,
            specular_alpha: specular_alpha,
        }
    }
    fn default() -> PhongMaterial {
        return PhongMaterial::new(1., 1., 1., 32.);
    }
}

struct PointLight {
    pos: Vector3<f64>,
    intensity: Vector3<f64>
}

struct Line {
    origin: Vector3<f64>,
    direction: Vector3<f64>,
}

impl Line {
    fn new(origin: Vector3<f64>, direction: Vector3<f64>) -> Line {
        return Line{
            origin: origin,
            direction: direction.normalize()
        };
    }
}

trait SceneObj  {
    fn intersects(&self, line: &Line) -> bool;
    fn get_intersections(&self, line: &Line) -> Vec<Vector3<f64>>;
}

trait PhongObj {
    fn normal(&self, point: &Vector3<f64>) -> Vector3<f64>;
    fn material<'a>(&'a self) -> &'a PhongMaterial;
}

struct PhongLighting {
    lights: Vec<PointLight>,
}

#[allow(non_snake_case)]
impl PhongLighting {
    fn get_right(&self, normal: &Vector3<f64>, light: &Vector3<f64>) -> Vector3<f64> {
        // R = 2 dot(L, N) N  - L
        let N = normal.normalize();
        let L = light.normalize();
        return (2. * dot(L, N) * N - L).normalize();
    }
    fn lighting(&self, cam: &Cam, point: &Vector3<f64>, obj: &PhongObj) -> Vector3<f64> {
        self.ambient(obj) + self.diffuse(point, obj) + self.specular(cam, point, obj)
    }
    fn ambient(&self, obj: &PhongObj) -> Vector3<f64> {
        self.lights.iter().map(|light| {
            let l_in = light.intensity;
            let k_ambient = obj.material().ambient;
            l_in * k_ambient
        }).fold(Vector3::zero(), |a, b| a + b)
    }
    fn diffuse(&self, point: &Vector3<f64>, obj: &PhongObj) -> Vector3<f64> {
        self.lights.iter().map(|light| {
            let l_in = light.intensity;
            let k_diffuse = obj.material().diffuse;
            let L = light.pos - point;
            let N = obj.normal(point);
            l_in * k_diffuse * dot(L, N)
        }).fold(Vector3::zero(), |a, b| a + b)
    }
    fn specular(&self, cam: &Cam, point: &Vector3<f64>, obj: &PhongObj) -> Vector3<f64> {
        self.lights.iter().map(|light| {
            let l_in = light.intensity;
            let k_specular = obj.material().specular;
            let alpha = obj.material().specular_alpha;
            let normalize_factor = (alpha + 2.) / (2. * PI);
            let N = obj.normal(point);
            let L = (light.pos - point).normalize();
            let R = self.get_right(&N, &L);
            let V = (cam.pos - point).normalize();
            let cos = dot(R, V).max(0.);
            l_in * k_specular * cos.powf(alpha) * normalize_factor
        }).fold(Vector3::zero(), |a, b| a + b)
    }
}

struct Sphere {
    pos: Vector3<f64>,
    radius: f64,
    material: PhongMaterial,
}

impl Sphere {
    fn determinate(&self, line: &Line) -> f64 {
        let l = line.direction;
        let o = line.origin;
        let c = self.pos;
        let r = self.radius;
        // dot(l, o - c)^2 - || o - c ||^2 + r^2
        return dot(l, o - c).powi(2) - (o - c).magnitude().powi(2) + r.powi(2);
    }
}

impl SceneObj for Sphere {
    fn intersects(&self, line: &Line) -> bool {
        return self.determinate(line) >= 0f64;
    }
    fn get_intersections(&self, line: &Line) -> Vec<Vector3<f64>> {
        let ldot = dot(line.direction, line.origin - self.pos);
        let det = self.determinate(line);

        if det < 0. {
            return vec!();
        } else if det == 0. {
            return vec!(line.origin + ldot * line.direction);
        } else {
            let d_minus = - ldot - det.sqrt();
            let d_plus =  - ldot + det.sqrt();
            return vec!(line.origin + line.direction*d_minus,
                        line.origin + line.direction*d_plus);
        }
    }
}
impl PhongObj for Sphere {
    fn material<'a>(&'a self) -> &'a PhongMaterial {
        &self.material
    }
    fn normal(&self, point: &Vector3<f64>) -> Vector3<f64> {
        point - self.pos
    }
}
struct Cam {
    pos: Vector3<f64>,
    orientation: Vector3<f64>,
}

impl Cam {
    fn new(pos: Vector3<f64>, orientation: Vector3<f64>) -> Self {
        return Cam {
            pos: pos,
            orientation: orientation.normalize()
        };
    }
}

struct Scene<T> {
    cam: Cam,
    scene_objs: Vec<Box<T>>,
    lighting: PhongLighting,
}


type IntersectionWithObject<'a, T> = Option<(Vector3<f64>, &'a Box<T>)>;


impl<T> Scene<T> where T: SceneObj + PhongObj{
    fn new(cam: Cam, scene_objs: Vec<Box<T>>, lighting: PhongLighting) -> Self {
        return Scene {
            cam: cam,
            scene_objs: scene_objs,
            lighting: lighting,
        }
    }
    fn objects_with_intersection<'a>(&'a self, line: &Line) -> Vec<&'a Box<T>> {
        return self.scene_objs.iter()
            .filter(|e| e.intersects(line))
            .collect();
    }
    fn nearest_intersection(&self, intersections: &Vec<Vector3<f64>>) -> Option<Vector3<f64>> {
        return intersections.iter()
            .map(|&x: &Vector3<f64>| (self.cam.pos.distance2(x), Some(x)))
            .fold((f64::INFINITY, None),
                |a, b| {
                    if a.0 < b.0 {
                        a
                    } else {
                        b
                    }
                }
            ).1
    }
    fn nearest_obj<'a>(&self,
                        a: IntersectionWithObject<'a, T>,
                        b: IntersectionWithObject<'a, T>) -> IntersectionWithObject<'a, T>  {
        match (a, b) {
            (Some((a_int, _)), Some((b_int, _))) =>
                if self.cam.pos.distance2(a_int) < self.cam.pos.distance2(b_int) {
                    a
                } else {
                    b
                },
            (Some(_), None) => a,
            (None, Some(_)) => b,
            _ => None,
        }
    }
    fn nearest_object_with_intersection<'a>(&'a self, line: &Line) -> Option<(Vector3<f64>, &'a Box<T>)> {
        let objs_intersects = self.objects_with_intersection(line);
        return objs_intersects.iter()
            .map(|o| (o.get_intersections(line), o))
            .map(|o| (self.nearest_intersection(&o.0).map(|i| (i, *o.1))))
            .fold((None), |a, b| self.nearest_obj(a, b));
    }

    fn render(self, rows: u32, cols: u32) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        return ImageBuffer::from_fn(cols, rows, |x, y| {
            let cols_h = cols as f64 / 2.;
            let rows_h = rows as f64 / 2.;
            let x_cam = (x as f64 - cols_h) / cols_h;
            let y_cam = (y as f64 - rows_h) / rows_h;
            let project_direction = Vector3::new(x_cam, y_cam, 0.) + self.cam.orientation;
            let line = Line::new(self.cam.pos, project_direction);
            let maybe_obj = self.nearest_object_with_intersection(&line);
            if let Some((nearest_intersection, s)) = maybe_obj {
                let light_vec = self.lighting.lighting(&self.cam, &nearest_intersection, &**s);
                Scene::<T>::vec_to_color(&light_vec)
            } else {
                image::Rgb([0u8, 0u8, 0u8])
            }
        });
    }
    fn vec_to_color(vec3: &Vector3<f64>) -> Rgb<u8> {
        let vec: &[f64; 3] = vec3.as_ref();
        let rgb = vec.iter()
            .map(|&v| 255.*clip(v, 0., 1.))
            .map(|v| v as u8)
            .collect::<Vec<_>>();
        return Rgb([rgb[0], rgb[1], rgb[2]]);
    }
}

fn clip(x: f64, a: f64, b: f64) -> f64{
     if x > b {
         b
     } else if x < a {
         a
     } else {
        x
     }
}

fn main() {
    let mut spheres = vec!();
    for i in 0..2 {
        for j in 0..2 {
            let sphere = Sphere{
                pos: Vector3::new(i as f64, -1.0, 2. + 2. * (j as f64)),
                radius: 0.45,
                material: PhongMaterial{
                    specular: 1.0,
                    diffuse: 1.8,
                    ambient: 0., // 2.2,
                    specular_alpha: 1000.,
                }
            };
            spheres.push(Box::new(sphere));
        }
    }

    let cam = Cam::new(Vector3::new(0., 0., -1.), Vector3::new(0., 0., -1.));

    let lighting = PhongLighting {
        lights: vec!(
            PointLight {
                pos: Vector3::new(-3., 4., -1.),
                intensity: Vector3::new(0.3, 0.01, 0.01),
            },
            PointLight {
                pos: Vector3::new(3., 4., -1.),
                intensity: Vector3::new(0.01, 0.01, 0.20),
            },
        )
    };
    let scene = Scene::new(cam, spheres, lighting);

    let img = scene.render(5000, 5000);
    // Write the contents of this image to the Writer in PNG format.
    let fout = Path::new("test.png");
    let _ = img.save(fout).unwrap();
}

#[test]
fn sphere_intersects_line() {
    let s = Sphere{
        pos: Vector3::new(0., 0., -2.),
        radius: 1.,
        material: PhongMaterial::default()
    };
    let l = Line::new(p, Vector3::new(0., 0., 1.));

    assert!(s.intersects(&l), "Line must intersect sphere");
    let intersections = s.get_intersections(&l);
    assert_eq!(intersections.len(), 2);
    let i0 = intersections[0];
    let i1 = intersections[1];
    assert_eq!(i0, Vector3::new(0., 0., -3.));
    assert_eq!(i1, Vector3::new(0., 0., -1.));
}