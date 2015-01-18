
kernel void partialTrueCount(global uchar *x, const uint size, const uint partSize, global uint *dest) {
    uint part = get_global_id(0);
    
    uint correct = 0;
    for (size_t i = part*partSize, end = min(part*partSize + partSize, size); i < end; ++i) {
        if (x[i]) {
            ++correct;
        }
    }
    dest[part] = correct;
}