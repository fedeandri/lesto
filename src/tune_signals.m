#import <Foundation/Foundation.h>

/* Low Power Mode as `lesto tune` reads it: 1 on, 0 off, -1 unknown. The tune pauses while it is on. */
int lesto_low_power_mode(void) {
    @autoreleasepool {
        NSProcessInfo *info = [NSProcessInfo processInfo];
        if (!info) return -1;
        if (![info respondsToSelector:@selector(isLowPowerModeEnabled)]) return -1;
        return info.lowPowerModeEnabled ? 1 : 0;
    }
}
