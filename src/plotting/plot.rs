use plotters::prelude::*;

pub fn plot_data(
    data: Vec<(f64, f64)>,
    caption: &str,
    x_desc: &str,
    y_desc: &str,
    save_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    //! Plot data using plotters
    //!
    //! ## Arguments:
    //! * `data`: `Vec<(f64, f64)>`, data to plot
    //! * `caption`: `&str`, caption for plot
    //! * `x_desc`: `&str`, description for x-axis
    //! * `y_desc`: `&str`, description for y-axis
    //! * `save_file`: `&str`, file to save plot to
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting data

    // create BitMapBackend
    let root = BitMapBackend::new(save_file, (1024, 768)).into_drawing_area();
    // fill the background with white
    root.fill(&WHITE)?;
    // find min and max values
    let min_x: f64 = data
        .iter()
        .map(|(x, _)| *x)
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max_x: f64 = data
        .iter()
        .map(|(x, _)| *x)
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let min_y: f64 = data
        .iter()
        .map(|(_, y)| *y)
        .min_by(|&x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max_y: f64 = data
        .iter()
        .map(|(_, y)| *y)
        .max_by(|&x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    // create chart
    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 120)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption(caption, ("sans-serif", 40))
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc(y_desc)
        .label_style(("sans-serif", 25))
        .axis_desc_style(("sans-serif", 25))
        .draw()?;

    // draw data
    chart.draw_series(LineSeries::new(data, &RED))?;

    // present data
    root.present().expect(
        "Unable to write result to file, please make sure 'plots' dir exists under current dir",
    );
    println!("Result has been saved to {}", save_file);
    Ok(())
}

pub fn scatter_data(
    data: Vec<(f64, f64)>,
    caption: &str,
    x_desc: &str,
    y_desc: &str,
    save_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    //! Scatters data using plotters
    //!
    //! ## Arguments:
    //! * `data`: `Vec<(f64, f64)>`, data to plot
    //! * `caption`: `&str`, caption for plot
    //! * `x_desc`: `&str`, description for x-axis
    //! * `y_desc`: `&str`, description for y-axis
    //! * `save_file`: `&str`, file to save plot to
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of scattering data

    // create BitMapBackend
    let root = BitMapBackend::new(save_file, (1024, 768)).into_drawing_area();
    // fill the background with white
    root.fill(&WHITE)?;
    // find min and max values
    let min_x: f64 = data
        .iter()
        .map(|(x, _)| *x)
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max_x: f64 = data
        .iter()
        .map(|(x, _)| *x)
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let min_y: f64 = data
        .iter()
        .map(|(_, y)| *y)
        .min_by(|&x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max_y: f64 = data
        .iter()
        .map(|(_, y)| *y)
        .max_by(|&x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    // create chart
    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 120)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption(caption, ("sans-serif", 40))
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc(y_desc)
        .label_style(("sans-serif", 25))
        .axis_desc_style(("sans-serif", 25))
        .draw()?;

    // scatter data
    chart.draw_series(data.iter().map(|point| Circle::new(*point, 3, &RED)))?;

    // present data
    root.present().expect(
        "Unable to write result to file, please make sure 'plots' dir exists under current dir",
    );
    println!("Result has been saved to {}", save_file);
    Ok(())
}
