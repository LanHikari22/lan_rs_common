#[cfg(feature = "incl_plot")]
mod plot {

}

/* 
use derive_builder::Builder;
#[allow(unused_imports)]
use im::{hashmap, hashset, ordset, vector, HashMap, HashSet, OrdSet, Vector};
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
#[allow(unused_imports)]
use rpds::{list, List};
// use libm;
// use log::Level;
// use rpds::{list, List};
// extern crate petgraph;
pub use crate::common::*;
use plotters::prelude::*;
use polars::prelude::*;
use polars::prelude::*;
use tap::prelude::*;

pub fn new_f64_lf_n_dim_multi<T>(
    lf: &LazyFrame,
    x_cols: &[&str],
    y_cols: &[&str],
    dim: usize,
    builder_fn: impl Fn(&LazyFrame, Vector<String>, Vector<String>) -> T,
) -> Result<T, String> {
    if y_cols.len() == 0 {
        Err(format!("There must be at least one y column"))
    } else if x_cols.len() != dim - 1 {
        Err(format!("There must be {dim} x column"))
    } else if x_cols.len() == 0 {
        Err(format!("There must be at least one column"))
    } else {
        let fields = {
            x_cols
                .into_iter()
                .map(|x_col| Field::new((*x_col).into(), DataType::Float64))
                .collect::<List<_>>()
                .pipe(|cols| {
                    y_cols //_
                        .into_iter()
                        .fold(cols, |acc, y_col| {
                            acc.push_front(Field::new((*y_col).into(), DataType::Float64))
                        })
                })
                .into_iter()
                .map(|field| field.clone())
                .collect::<Vec<Field>>()
        };

        let schema = //_
            Schema::from_iter(fields);

        if let Err(e) = validate_lf_with_schema(lf, schema) {
            Err(e)
        } else {
            Ok(builder_fn(
                lf,
                x_cols
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect::<Vector<_>>(),
                y_cols
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect::<Vector<_>>(),
            ))
        }
    }
}

#[derive(Clone)]
pub struct F32LazyframeNDimMulti {
    pub lf: LazyFrame,
    pub x_cols: Vector<String>,
    pub y_cols: Vector<String>,
}

impl F32LazyframeNDimMulti {
    pub fn new(lf: &LazyFrame, x_cols: &[&str], y_cols: &[&str]) -> Result<Self, String> {
        let builder_fn = {
            |lf: &LazyFrame, x_cols: Vector<String>, y_cols: Vector<String>| {
                //_
                Self {
                    lf: lf.clone(),
                    x_cols,
                    y_cols,
                }
            }
        };

        new_f64_lf_n_dim_multi(lf, x_cols, y_cols, x_cols.len() + 1, builder_fn)
    }

    /// Similar to new, but does further processing first on the df to ensure
    /// 1. for all x, there is a unique y (group y by x and sum if `agg_method` == "sum", ... etc)
    /// Possible agg_method values are ["sum", "mean", "count"]
    pub fn process(
        lf: &LazyFrame,
        x_cols: &[&str],
        y_cols: &[&str],
        agg_method: &str,
    ) -> Result<Self, String> {
        let lf_agg = {
            let enclosed_x_cols = {
                x_cols //_
                    .into_iter()
                    .map(|x_col_name| col(*x_col_name))
                    .collect::<Vec<_>>()
            };

            let group = {
                lf //_
                    .clone()
                    .group_by(enclosed_x_cols)
            };

            let agg_fn = {
                |col: Expr| match agg_method {
                    "sum" => Ok(col.sum()),
                    "mean" => Ok(col.mean()),
                    "count" => Ok(col.count()),
                    other => Err(format!("Unsupported agg_method {other}")),
                }
            };

            let enclosed_y_cols = {
                y_cols //_
                    .into_iter()
                    .map(|col_name| agg_fn(col(*col_name)))
                    .collect::<Result<Vec<_>, String>>()?
            };

            group //_
                .agg(enclosed_y_cols)
        };

        Self::new(&lf_agg, x_cols, y_cols)
    }

    pub fn get_xs_y_points_per_y(&self) -> Result<Vec<Vec<(Vec<f64>, f64)>>, String> {
        Self::to_xs_y_points_per_y(&self.lf, &self.x_cols, &self.y_cols)
    }

    pub fn to_xs_y_points_per_y(
        lf: &LazyFrame,
        x_col_names: &Vector<String>,
        y_col_names: &Vector<String>,
    ) -> Result<Vec<Vec<(Vec<f64>, f64)>>, String> {
        let df = {
            let selected_cols = {
                let l = {
                    x_col_names
                        .iter()
                        .fold(List::new(), |cols, x_col_name| {
                            cols.push_front(col(x_col_name))
                        })
                        .into_iter()
                        .map(|field| field.clone())
                        .collect::<List<_>>()
                };

                y_col_names
                    .iter()
                    .fold(l, |cols, y_col_name| cols.push_front(col(y_col_name)))
                    .into_iter()
                    .map(|field| field.clone())
                    .collect::<Vec<_>>()
            };

            lf //_
                .clone()
                .select(&selected_cols)
                .collect()
                .map_err(|e| e.to_string())?
        };

        let col_names_to_cols_fn = {
            |col_names: &Vector<String>| {
                col_names
                    .iter()
                    .map(|col| {
                        df //_
                            .column(col)
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            }
        };

        let x_cols = //_
            col_names_to_cols_fn(x_col_names)?;
        let y_cols = //_
            col_names_to_cols_fn(y_col_names)?;

        y_cols
            .iter()
            .map(|y_col| {
                // So for each column we need a Vec<(Vec<f64>, f64)> because these are points in
                // n-dimensional space, we're just making a distinction between their meaning, ie
                // ([x0, x1, ..., xN-1], y)
                (0..y_col.len())
                    .map(|sample| {
                        let get_val_fn = {
                            |is_x: bool, which: usize, col: &Column| -> Result<f64, String> {
                                col.get(sample)
                                    .map_err(|e| e.to_string())?
                                    .pipe(|val| {
                                        //_
                                        match val {
                                            AnyValue::Float64(f) => Ok(f),
                                            typ => {
                                                //_
                                                if is_x {
                                                    format!("x_col[{which}]")
                                                } else {
                                                    "y_col".to_string()
                                                }
                                                .pipe(|s| {
                                                    Err(format!(
                                                        "{s}: Expected Float64 but found {}",
                                                        typ.dtype()
                                                    ))
                                                })
                                            }
                                        }
                                    })?
                                    .pipe(|res| Ok(res))
                            }
                        };

                        let x_vals = {
                            x_cols
                                .iter()
                                .enumerate()
                                .map(|(i, x_col)| get_val_fn(true, i, x_col))
                                .collect::<Result<Vec<_>, String>>()?
                        };
                        let y_val = //_
                            get_val_fn(false, 0, y_col)?;

                        Ok((x_vals, y_val))
                    })
                    .collect::<Result<Vec<(Vec<f64>, f64)>, String>>()
            })
            .collect::<Result<Vec<Vec<_>>, String>>()
    }

    pub fn get_min_max_for_col_f64(col: &Vec<f64>, col_name: &str) -> Result<(f64, f64), String> {
        col //_
            .iter()
            .minmax()
            .pipe(|min_max_res| {
                //_
                match min_max_res {
                    itertools::MinMaxResult::MinMax(min, max) => Ok((*min, *max)),
                    itertools::MinMaxResult::OneElement(minmax) => Ok((*minmax, *minmax)),
                    // itertools::MinMaxResult::NoElements => todo!(),
                    _ => Err(format!("{col_name} needs to have at least 1 element"))
                }
            })
    }

    pub fn to_xs_y_ranges(
        xs_y_points_per_y: &Vec<Vec<(Vec<f64>, f64)>>,
    ) -> Result<(Vector<(f64, f64)>, (f64, f64)), String> {
        xs_y_points_per_y
            .iter()
            .map(|xs_y_points| {
                let (x_min, x_max) = //_
                    Self::get_min_max_for_xy_points(xs_y_points, true)?;

                let x_ranges = { xs_y_points.iter().enumerate().map(|(sample, (xs, y))| {}) };

                let (y_min, y_max) = //_
                    Self::get_min_max_for_xs_y_points(xs_y_points, false, 0)?;

                Ok(((x_min, x_max), (y_min, y_max)))
                // Ok(((x_min..x_max), (y_min..y_max)))
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .fold(None, |acc, ((x_min, x_max), (y_min, y_max))| {
                if let Some(((prev_x_min, prev_x_max), (prev_y_min, prev_y_max))) = acc {
                    let find_new_minmax_fn = //_
                        |cur_min, prev_min, cur_max, prev_max| {
                            (
                                //_
                                f64::min(cur_min, prev_min),
                                f64::max(cur_max, prev_max),
                            )
                        };

                    let (new_x_min, new_x_max) = //_
                        find_new_minmax_fn(x_min, prev_x_min, x_max, prev_x_max);
                    let (new_y_min, new_y_max) = //_
                        find_new_minmax_fn(y_min, prev_y_min, y_max, prev_y_max);

                    Some(((new_x_min, new_x_max), (new_y_min, new_y_max)))
                } else {
                    Some(((x_min, x_max), (y_min, y_max)))
                }
            })
            .pipe(|opt| match opt {
                Some(opt) => Ok(opt),
                None => Err(format!("Expected to find min max but found none")),
            })?
            .pipe(|((x_min, x_max), (y_min, y_max))| Ok(((x_min..x_max), (y_min..y_max))));
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct F32LazyframeNDim {
    pub lf: LazyFrame,
    pub x_cols: Vector<String>,
    pub y_col: String,
}

impl F32LazyframeNDim {
    pub fn new(lf: &LazyFrame, x_cols: &[&str], y_col: &str) -> Result<Self, String> {
        let builder_fn = {
            |lf: &LazyFrame, x_cols: Vector<String>, y_cols: Vector<String>| {
                //_
                Self {
                    lf: lf.clone(),
                    x_cols,
                    y_col: y_cols[0].clone(),
                }
            }
        };

        new_f64_lf_n_dim_multi(lf, x_cols, &[y_col], x_cols.len() + 1, builder_fn)
    }
}

#[derive(Clone)]
pub struct F64Lazyframe2DMulti {
    pub lf: LazyFrame,
    pub x_col: String,
    pub y_cols: Vector<String>,
}

impl F64Lazyframe2DMulti {
    pub fn new(lf: &LazyFrame, x_col: &str, y_cols: &[&str]) -> Result<Self, String> {
        let builder_fn = //_
            |lf: &LazyFrame, x_cols: Vector<String>, y_cols: Vector<String>| {
                Self {
                    lf: lf.clone(),
                    x_col: x_cols[0].clone(),
                    y_cols,
                }
            };

        new_f64_lf_n_dim_multi(lf, &vec![x_col], y_cols, 2, builder_fn)
    }

    /// Similar to new, but does further processing first on the df to ensure
    /// 1. for all x, there is a unique y (group y by x and sum if `agg_method` == "sum", ... etc)
    /// Possible agg_method values are ["sum", "mean", "count"]
    pub fn process(
        lf: &LazyFrame,
        x_col: &str,
        y_cols: &[&str],
        agg_method: &str,
    ) -> Result<Self, String> {
        let lf_agg = {
            let group = {
                lf //_
                    .clone()
                    .group_by([x_col])
            };

            let agg_fn = |col: Expr| match agg_method {
                "sum" => Ok(col.sum()),
                "mean" => Ok(col.mean()),
                "count" => Ok(col.count()),
                other => Err(format!("Unsupported agg_method {other}")),
            };

            let enclosed_y_cols = {
                y_cols //_
                    .into_iter()
                    .map(|col_name| agg_fn(col(*col_name)))
                    .collect::<Result<Vec<_>, String>>()?
            };

            group //_
                .agg(enclosed_y_cols)
        };

        Self::new(&lf_agg, x_col, y_cols)
    }

    pub fn get_xy_points(&self) -> Result<Vec<Vec<(f64, f64)>>, String> {
        Self::to_xy_points(&self.lf, &self.x_col, &self.y_cols)
    }

    pub fn to_xy_points(
        lf: &LazyFrame,
        x_col_name: &str,
        y_col_names: &Vector<String>,
    ) -> Result<Vec<Vec<(f64, f64)>>, String> {
        let df = {
            let selected_cols = {
                let l = //_
                    List::new().push_front(col(x_col_name));

                y_col_names
                    .iter()
                    .fold(l, |cols, y_col_name| cols.push_front(col(y_col_name)))
                    .into_iter()
                    .map(|field| field.clone())
                    .collect::<Vec<_>>()
            };

            lf //_
                .clone()
                .select(&selected_cols)
                .collect()
                .map_err(|e| e.to_string())?
        };

        let x_col = {
            df //_
                .column(x_col_name)
                .map_err(|e| e.to_string())?
        };

        let y_cols = {
            y_col_names
                .iter()
                .map(|y_col| {
                    df //_
                        .column(y_col)
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        y_cols
            .into_iter()
            .map(|y_col| {
                (0..x_col.len())
                    .map(|i| {
                        (
                            x_col
                                .get(i)
                                .map_err(|e| e.to_string())?
                                .pipe(|val| match val {
                                    AnyValue::Float64(f) => Ok(f),
                                    typ => Err(format!(
                                        "x_col: Expected Float64 but found {}",
                                        typ.dtype()
                                    )),
                                })?,
                            y_col
                                .get(i)
                                .map_err(|e| e.to_string())?
                                .pipe(|val| match val {
                                    AnyValue::Float64(f) => Ok(f),
                                    typ => Err(format!(
                                        "y_col: Expected Float64 but found {}",
                                        typ.dtype()
                                    )),
                                })?,
                        )
                            .pipe(|res| Ok(res))
                    })
                    .collect::<Result<Vec<_>, String>>()?
                    .pipe(|vec| Ok(vec))
            })
            .collect::<Result<Vec<Vec<_>>, String>>()
    }

    pub fn get_min_max(
        xy_points_per_y: &Vec<(f64, f64)>,
        is_x: bool,
    ) -> Result<(f64, f64), String> {
        xy_points_per_y //_
            .iter()
            .map(|(x, y)| if is_x { x } else { y })
            .minmax()
            .pipe(|min_max_res| match min_max_res {
                itertools::MinMaxResult::MinMax(min, max) => Ok((*min, *max)),
                _ => if is_x { "x_col" } else { "y_col" }
                    .pipe(|s| Err(format!("{s} needs to have at least 3 elements"))),
            })
    }

    pub fn to_xy_ranges(
        xy_points: &Vec<Vec<(f64, f64)>>,
    ) -> Result<(std::ops::Range<f64>, std::ops::Range<f64>), String> {
        xy_points
            .iter()
            .map(|xy_points_per_y| {
                let (x_min, x_max) = //_
                    Self::get_min_max(xy_points_per_y, true)?;

                let (y_min, y_max) = //_
                    Self::get_min_max(xy_points_per_y, false)?;

                Ok(((x_min, x_max), (y_min, y_max)))
                // Ok(((x_min..x_max), (y_min..y_max)))
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .fold(None, |acc, ((x_min, x_max), (y_min, y_max))| {
                if let Some(((prev_x_min, prev_x_max), (prev_y_min, prev_y_max))) = acc {
                    let find_new_minmax_fn = //_
                        |cur_min, prev_min, cur_max, prev_max| {
                            (
                                //_
                                f64::min(cur_min, prev_min),
                                f64::max(cur_max, prev_max),
                            )
                        };

                    let (new_x_min, new_x_max) = //_
                        find_new_minmax_fn(x_min, prev_x_min, x_max, prev_x_max);
                    let (new_y_min, new_y_max) = //_
                        find_new_minmax_fn(y_min, prev_y_min, y_max, prev_y_max);

                    Some(((new_x_min, new_x_max), (new_y_min, new_y_max)))
                } else {
                    Some(((x_min, x_max), (y_min, y_max)))
                }
            })
            .pipe(|opt| match opt {
                Some(opt) => Ok(opt),
                None => Err(format!("Expected to find min max but found none")),
            })?
            .pipe(|((x_min, x_max), (y_min, y_max))| Ok(((x_min..x_max), (y_min..y_max))))
    }
}

#[derive(Clone)]
pub struct F64Lazyframe2D {
    pub lf: LazyFrame,
    pub x_col: String,
    pub y_col: String,
}

impl F64Lazyframe2D {
    pub fn new(lf: &LazyFrame, x_col: &str, y_col: &str) -> Result<Self, String> {
        let builder_fn = {
            |lf: &LazyFrame, x_cols: Vector<String>, y_cols: Vector<String>| {
                //_
                Self {
                    lf: lf.clone(),
                    x_col: x_cols[0].clone(),
                    y_col: y_cols[0].clone(),
                }
            }
        };

        new_f64_lf_n_dim_multi(lf, &vec![x_col], &[y_col], 2, builder_fn)
    }

    /// Similar to new, but does further processing first on the df to ensure
    /// 1. for all x, there is a unique y (group y by x and sum if `agg_method` == "sum", ... etc)
    /// Possible agg_method values are ["sum", "mean", "count"]
    pub fn process(
        lf: &LazyFrame,
        x_col: &str,
        y_col: &str,
        agg_method: &str,
    ) -> Result<Self, String> {
        let F64Lazyframe2DMulti { lf, x_col, y_cols } = //_
            F64Lazyframe2DMulti::process(lf, x_col, &[y_col], agg_method)?;

        Self::new(&lf, &x_col, &y_cols[0].clone())
    }

    pub fn get_xy_points(&self) -> Result<Vec<(f64, f64)>, String> {
        F64Lazyframe2DMulti::to_xy_points(&self.lf, &self.x_col, &vector![self.y_col.clone()])
            .and_then(|vec| Ok(vec.into_iter().next().unwrap()))
    }

    fn to_xy_ranges(
        xy_points: &Vec<(f64, f64)>,
    ) -> Result<(std::ops::Range<f64>, std::ops::Range<f64>), String> {
        let (x_min, x_max) = //_
            F64Lazyframe2DMulti::get_min_max(xy_points, true)?;
        let (y_min, y_max) = //_
            F64Lazyframe2DMulti::get_min_max(xy_points, false)?;

        Ok(((x_min..x_max), (y_min..y_max)))
    }
}

#[derive(Clone)]
pub struct F32LazyFrame3DMulti {
    pub lf: LazyFrame,
    pub x_cols: [String; 2],
    pub y_cols: Vector<String>,
}

impl F32LazyFrame3DMulti {
    pub fn new(lf: &LazyFrame, x_cols: &[&str; 2], y_cols: &[&str]) -> Result<Self, String> {
        let builder_fn = {
            |lf: &LazyFrame, x_cols: Vector<String>, y_cols: Vector<String>| {
                //_
                Self {
                    lf: lf.clone(),
                    x_cols: [x_cols[0].clone(), x_cols[1].clone()],
                    y_cols,
                }
            }
        };

        new_f64_lf_n_dim_multi(lf, x_cols, y_cols, x_cols.len() + 1, builder_fn)
    }
}

#[derive(Clone)]
pub struct F32LazyFrame3D {
    pub lf: LazyFrame,
    pub x_cols: [String; 2],
    pub y_col: String,
}

impl F32LazyFrame3D {
    pub fn new(lf: &LazyFrame, x_cols: &[&str; 2], y_col: &str) -> Result<Self, String> {
        let builder_fn = {
            |lf: &LazyFrame, x_cols: Vector<String>, y_cols: Vector<String>| {
                //_
                Self {
                    lf: lf.clone(),
                    x_cols: [x_cols[0].clone(), x_cols[1].clone()],
                    y_col: y_cols[0].clone(),
                }
            }
        };

        new_f64_lf_n_dim_multi(lf, x_cols, &[y_col], x_cols.len() + 1, builder_fn)
    }
}

pub fn create_bitmap_backend_drawing_area(
    png_path: &str,
    w: u32,
    h: u32,
) -> DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> {
    /* w: 640, h: 480 */
    BitMapBackend::new(png_path, (w, h)) //_
        .into_drawing_area()
}

#[derive(Clone, Builder)]
pub struct PlottersScatter2Options {
    /// opt_caption_label takes presedence over this option
    #[builder(default = "false")]
    set_caption_as_x_vs_y: bool,
    #[builder(default = "None")]
    opt_caption_label: Option<String>,
    #[builder(default = "\"sans-serif\"")]
    caption_font: &'static str,
    #[builder(default = "25")]
    caption_font_size: u32,
    #[builder(default = "true")]
    do_label_x_axis: bool,
    #[builder(default = "true")]
    do_label_y_axis: bool,

    #[builder(default = "10")]
    margin: u32,

    #[builder(default = "30")]
    x_label_area_size: u32,
    #[builder(default = "30")]
    y_label_area_size: u32,

    #[builder(default = "3")]
    datapoint_circle_size: u32,
    #[builder(default = "RED.filled()")]
    datapoint_circle_style: ShapeStyle,

    /// opt_label takes presedence over this option
    #[builder(default = "true")]
    incl_label_as_y_col: bool,
    #[builder(default = "None")]
    opt_label: Option<String>,
}

pub fn plotters_scatter2<DB: DrawingBackend>(
    lf: &F64Lazyframe2D,
    root: &DrawingArea<DB, plotters::coord::Shift>,
    options: Option<PlottersScatter2Options>,
) -> Result<(), String> {
    let options = match options {
        Some(options) => options,
        None => PlottersScatter2OptionsBuilder::default().build().unwrap(),
    };

    root //_
        .fill(&WHITE)
        .map_err(|e| e.to_string())?;

    let xy_points = //_
        lf.to_xy_points()?;

    let (df_x_range, df_y_range) = //_
        F64Lazyframe2D::to_xy_ranges(&xy_points)?;

    let mut chart = {
        ChartBuilder::on(root)
            .pipe_borrow_mut(|builder| {
                if let Some(caption) = options.opt_caption_label {
                    builder //_
                        .caption(
                            &caption,
                            (options.caption_font, options.caption_font_size).into_font(),
                        )
                } else if options.set_caption_as_x_vs_y {
                    builder //_
                        .caption(
                            &format!("{} vs {}", lf.x_col, lf.y_col),
                            (options.caption_font, options.caption_font_size).into_font(),
                        )
                } else {
                    builder
                }
            })
            .margin(options.margin)
            .x_label_area_size(options.x_label_area_size)
            .y_label_area_size(options.y_label_area_size)
            .build_cartesian_2d(df_x_range, df_y_range)
            .map_err(|e| e.to_string())?
    };

    chart //_
        .configure_mesh()
        .pipe_borrow_mut(|mesh| {
            if options.do_label_x_axis {
                mesh.x_desc(lf.x_col.clone());
            }

            if options.do_label_y_axis {
                mesh.y_desc(lf.y_col.clone());
            }

            mesh
        })
        .draw()
        .map_err(|e| e.to_string())?;

    let added_label = {
        chart //_
            // .draw_series(LineSeries::new(xy_points, &RED))
            .draw_series({
                xy_points //_
                    .into_iter()
                    .map(|(x, y)| {
                        Circle::new(
                            (x, y),
                            options.datapoint_circle_size,
                            options.datapoint_circle_style,
                        )
                    })
            })
            .map_err(|e| e.to_string())?
            .pipe(|series| {
                let (added_label, series) = {
                    if let Some(label) = options.opt_label {
                        (
                            true,
                            series //_
                                .label(&label),
                        )
                    } else if options.incl_label_as_y_col {
                        (
                            true,
                            series //_
                                .label(&lf.y_col),
                        )
                    } else {
                        (false, series)
                    }
                };

                if added_label {
                    series //_
                        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
                }

                added_label
            })
    };

    if added_label {
        chart //_
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .map_err(|e| e.to_string())?;
    }

    root //_
        .present()
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_plotters_scatter2() {
        let manual_test_enabled = //_
            true;

        if manual_test_enabled {
            let csv_lf = {
                load_csv_into_lazyframe(
                    // "/home/lan/data/kaggle/datasets/wisam1985/advanced-soybean-agricultural-dataset-2025/Advanced Soybean Agricultural Dataset.csv"
                    "/home/lan/data/kaggle/datasets/uciml/iris/Iris.csv",
                )
                .unwrap()
            };

            let lf = {
                // F64Lazyframe2D::process(
                //     //_
                //     &csv_lf,
                //     "Number of Pods (NP)",
                //     "Plant Height (PH)",
                //     "mean",
                // )
                F64Lazyframe2D::process(
                    //_
                    &csv_lf,
                    // "Number of Pods (NP)",
                    // "Plant Height (PH)",
                    "SepalLengthCm",
                    "SepalWidthCm",
                    "mean",
                )
                .unwrap()
            };

            println!("{}", lf.clone().lf.collect().unwrap());

            save_lf_to_csv(lf.lf.clone(), "lf.csv").unwrap();

            create_bitmap_backend_drawing_area("test.png", 1 * 640, 1 * 480)
                .pipe(|root| {
                    plotters_scatter2(
                        &lf,
                        &root,
                        Some(
                            PlottersScatter2OptionsBuilder::default()
                                .opt_caption_label(Some("advanced soybean graph".to_string()))
                                .margin(30)
                                .build()
                                .unwrap(),
                        ),
                    )
                })
                .unwrap();

            panic!(
                "{}",
                str_style("Manual test executed succssfully", AnsiStyles::Green)
            );
        }
    }
}

*/