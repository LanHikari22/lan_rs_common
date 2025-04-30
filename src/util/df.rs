/* \begin{dataframe} */

#[cfg(feature = "df")]
pub fn validate_lf_with_schema(lf: &LazyFrame, schema: Schema) -> Result<(), String> {
    schema
        .iter_fields()
        .fold_while(Ok(()), |_, field| {
            let checks = {
                lf //_
                    .clone()
                    .collect_schema()
                    .map_err(|e| e.to_string())
                    .and_then(|lf_schema| {
                        if lf_schema.contains(field.name()) {
                            Ok(lf_schema)
                        } else {
                            Err(format!("Failed to validate schema: Could not find column {} in df", field.name()))
                        }
                    })
                    .and_then(|lf_schema| {
                        Ok(lf_schema.get_field(field.name()).unwrap())
                    })
                    .and_then(|lf_field| {
                        if lf_field.dtype == *field.dtype() {
                            Ok(field)
                        } else {
                            Err(format!("Failed to validate schema: Expected {} but found {}", lf_field.dtype, field.dtype()))
                        }
                    })
                    .and_then(|_| Ok(()))
            };

            if let Err(e) = checks {
                Done(Err(e))
            } else {
                Continue(Ok(()))
            }
        }).into_inner()
}

#[cfg(feature = "df")]
pub fn validate_df_with_schema(df: &DataFrame, schema: Schema) -> Result<(), String> {
    schema
        .iter_fields()
        .fold_while(Ok(()), |_, field| {
            let col_res = {
                df //_
                    .column(field.name())
                    .map_err(|_| format!("Failed to validate schema: Could not find column {} in df", field.name()))
            };

            if let Ok(col) = col_res {
                if col.dtype() != field.dtype() {
                    Done(Err(format!("Failed to validate schema: Expected {} but found {}", col.dtype(), field.dtype())))
                } else {
                    Continue(Ok(()))
                }
            } else {
                Done(Err(col_res.unwrap_err()))
            }
        }).into_inner()
}

#[cfg(feature = "df")]
pub fn read_csv_into_dataframe(csv_path: &str) -> Result<DataFrame, String> {
    let file = {
        File::open(csv_path)
            .map_err(|e| e.to_string())?
    };

    let df = {
        CsvReader::new(file)
            .finish()
            .map_err(|e| e.to_string())?
    };

    Ok(df)
}

#[cfg(feature = "df")]
pub fn load_csv_into_lazyframe(csv_path: &str) -> Result<LazyFrame, String> {
    let df = {
        LazyCsvReader::new(csv_path)
            .finish()
            .map_err(|e| e.to_string())?
    };

    Ok(df)
}

#[cfg(feature = "df")]
pub fn save_lf_to_csv(lf: LazyFrame, csv_path: &str) -> Result<(), String> {
    let file = {
        File::create(csv_path)
            .map_err(|e| e.to_string())?
    };

    let mut df = {
        lf //_
            .collect()
            .map_err(|e| e.to_string())?
    };

    CsvWriter::new(file) //_
        .finish(&mut df)
        .map_err(|e| e.to_string())?
        .pipe(|_| Ok(()))
}

/* \end{dataframe} */