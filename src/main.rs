extern crate cgmath;
extern crate image;

use std::vec::Vec;
use std::f64;
use std::path::Path;
use image::ImageBuffer;
use cgmath::{Vector3, InnerSpace, MetricSpace, dot};


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

struct Sphere {
    pos: Vector3<f64>,
    radius: f64,
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

struct Scene {
    cam: Cam,
    scene_objs: Vec<Box<SceneObj>>
}

fn min_f64(vec: &Vec<f64>) -> f64 {
    return vec.iter().cloned().fold(f64::NAN, f64::min);
}

impl Scene {
    fn new(cam: Cam, scene_objs: Vec<Box<SceneObj>>) -> Self {
        return Scene {
            cam: cam,
            scene_objs: scene_objs,
        }
    }
    fn objects_with_intersection<'a>(&'a self, line: &Line) -> Vec<&'a Box<SceneObj>> {
        return self.scene_objs.iter()
            .filter(|e| e.intersects(line))
            .collect();
    }

    fn nearest_object_with_intersection<'a>(&'a self, line: &Line) -> Option<&'a Box<SceneObj>> {
        let objs_intersects = self.objects_with_intersection(line);
        let (_, maybe_obj) = objs_intersects.iter()
            .map(|o| (o.get_intersections(line), o))
            .filter(|o| o.0.len() >= 1)
            .map(|o| (min_f64(&o.0.iter()
                            .map(|&x| self.cam.pos.distance2(x))
                            .collect())
                      , Some(o.1)))
            .fold((f64::NAN, None), |a, b| {
                if a.0 < b.0 {
                    a
                } else {
                    b
                }
            });
        if let Some(obj) = maybe_obj {
            return Some(*obj);
        } else {
            return None;
        }
    }

    fn render(self, rows: u32, cols: u32) -> ImageBuffer<image::Luma<u8>, Vec<u8>> {
        let project_direction = self.cam.orientation;
        return ImageBuffer::from_fn(cols, rows, |x, y| {
            let cols_h = cols as f64 / 2.;
            let rows_h = rows as f64 / 2.;
            let x_cam = (x as f64 - cols_h) / cols_h;
            let y_cam = (y as f64 - rows_h) / rows_h;
            let line = Line::new(Vector3::new(x_cam, y_cam, 0.), project_direction);

            let maybe_obj = self.nearest_object_with_intersection(&line);
            if let Some(s) = maybe_obj {
                let z = s.get_intersections(&line).iter()
                    .map(|p| p.z)
                    .fold(f64::NAN, f64::min);
                image::Luma([((z - 2.).abs() * 100. + 25.) as u8])
            } else {
                image::Luma([0u8])
            }
        });
    }
}

fn main() {
    let sphere = Sphere{pos: Vector3::new(0., 0., 2.), radius: 0.5};

    let cam = Cam::new(Vector3::new(0., 0., -1.), Vector3::new(0., 0., -1.));

    let scene = Scene::new(cam, vec!(Box::new(sphere)));

    let img = scene.render(512, 512);
    // Write the contents of this image to the Writer in PNG format.
    let fout = Path::new("test.png");
    let _ = img.save(fout).unwrap();
}

#[test]
fn sphere_intersects_line() {
    let s = Sphere{pos: Vector3::new(0., 0., -2.), radius: 1.};
    let l = Line::new(p, Vector3::new(0., 0., 1.));

    assert!(s.intersects(&l), "Line must intersect sphere");
    let intersections = s.get_intersections(&l);
    assert_eq!(intersections.len(), 2);
    let i0 = intersections[0];
    let i1 = intersections[1];
    assert_eq!(i0, Vector3::new(0., 0., -3.));
    assert_eq!(i1, Vector3::new(0., 0., -1.));
}