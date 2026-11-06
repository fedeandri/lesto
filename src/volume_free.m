#import <Foundation/Foundation.h>
#include <stdint.h>

/* Bytes the OS will grant a write at `path`: statfs/NSFileSystemFreeSize
 * exclude purgeable space, which macOS releases on demand (a volume "36 GB
 * free" by df had 117 GB for important usage). 0 = probe failed. */
uint64_t msv_volume_free_for_use(const char *path) {
    if (!path) return 0;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSNumber *avail = nil;
    if (![url getResourceValue:&avail forKey:NSURLVolumeAvailableCapacityForImportantUsageKey error:nil] || !avail)
        return 0;
    return avail.unsignedLongLongValue;
}
