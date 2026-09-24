#ifndef CONFIG_H
#define CONFIG_H

#define WUNJO_VERSION "@WUNJO_VERSION_STRING@"
#define WUNJO_FULL_VERSION_STRING "@WUNJO_FULL_VERSION_STRING@"

/* Date this build was configured, ISO 8601. What the update check actually
   compares: a release is newer than this build or it is not, and that stays
   true however the version is numbered, re-tagged or re-numbered later. */
#define WUNJO_BUILD_DATE "@WUNJO_BUILD_DATE@"

#define MLT_PREFIX "@MLT_PREFIX@"
#define MLT_MIN_MAJOR_VERSION @MLT_MIN_MAJOR_VERSION@
#define MLT_MIN_MINOR_VERSION @MLT_MIN_MINOR_VERSION@
#define MLT_MIN_PATCH_VERSION @MLT_MIN_PATCH_VERSION@

#define FFMPEG_SUFFIX "@FFMPEG_SUFFIX@"

#cmakedefine HAVE_MALLOC_H 1
#cmakedefine HAVE_PTHREAD_H 1

#endif
