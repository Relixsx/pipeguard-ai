# Original PipeGuard application

This directory preserves the original backend, frontend, data, trained weights and requirements from source commit `f3acfde210161a396cb1835bcabd2a413ba20e78`.

It is an archival baseline. It is excluded from the current Docker image and is not served by the research demo. See the original-source audit and the main README before interpreting its evaluation or using its label-driven demonstration paths.
# Archived credential handling

The archived server requires explicit `SECRET_KEY` and `ADMIN_PASSWORD` environment variables if run separately. Embedded credential defaults and password logging have been removed. Historical runtime logs are excluded from the update.
