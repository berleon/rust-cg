extern crate cgmath;
extern crate image;

use std::cmp;
use std::vec::Vec;
use std::f64;
use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;
use image::ImageBuffer;
use cgmath::{Vector3, InnerSpace, dot};


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

trait SceneElem {
    fn intersects(&self, line: &Line) -> bool;
    fn getIntersections(&self, line: &Line) -> Vec<Vector3<f64>>;
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
impl SceneElem for Sphere {
    fn intersects(&self, line: &Line) -> bool {
        return self.determinate(line) >= 0f64;
    }
    fn getIntersections(&self, line: &Line) -> Vec<Vector3<f64>> {
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


struct Scene<T: SceneElem> {
    cam: Cam,
    scene_elems: Vec<Box<T>>,
}


fn main() {
    let p = Vector3::new(0f64, 0f64, 0f64);
    let s = Sphere{pos: Vector3::new(0f64, 0f64, -2f64), radius: 0.5};

    let l = Line::new(p, Vector3::new(0f64, 0f64, 1f64));

    println!("Does line intersect sphere: {}", s.intersects(&l));
    let intersections = s.getIntersections(&l);
    let i0 = intersections[0];
    let i1 = intersections[1];
    println!("Line intersect sphere at: {}, {}, {}", i0[0], i0[1], i0[2]);
    println!("Line intersect sphere at: {}, {}, {}", i1[0], i1[1], i1[2]);

    //Construct a new by repeated calls to the supplied closure.
    let rows = 512;
    let cols = 512;
    let project_direction = Vector3::new(0., 0., -1.);
    let img = ImageBuffer::from_fn(cols, rows, |x, y| {
        let cols_h = cols as f64 / 2.;
        let rows_h = rows as f64 / 2.;
        let x_cam = (x as f64 - cols_h) / cols_h;
        let y_cam = (y as f64 - rows_h) / rows_h;
        let line = Line::new(Vector3::new(x_cam, y_cam, 0.), project_direction);
        if s.intersects(&line) {
            let z = s.getIntersections(&line).iter()
                .map(|p| p.z)
                .fold(f64::NAN, f64::min);
            image::Luma([((z - s.pos.z).abs() / s.radius * 230. + 25.) as u8])
        } else {
            image::Luma([0u8])
        }
    });
    // Write the contents of this image to the Writer in PNG format.
    let fout = Path::new("test.png");
    let _ = img.save(fout).unwrap();
}