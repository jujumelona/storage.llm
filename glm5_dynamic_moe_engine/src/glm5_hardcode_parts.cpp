#include "glm5_pc_engine.h"

static const glm5_storage_part_spec_t kGlm5Parts[] = {
    {1,"parts/glm5.1-storage-part01.juju",23643820557ull,8499,931,0,120,4036552192ull,19603062976ull,4036552192ull,1,4,1,5},
    {2,"parts/glm5.1-storage-part02.juju",19369187509ull,7806,861,0,57,1282371072ull,18083020992ull,1282371072ull,5,4,4,6},
    {3,"parts/glm5.1-storage-part03.juju",23176814429ull,9292,1024,0,76,1709828096ull,21462384832ull,1709828096ull,9,4,8,6},
    {4,"parts/glm5.1-storage-part04.juju",21367618930ull,8527,939,0,76,1709828096ull,19653525696ull,1709828096ull,13,4,12,6},
    {5,"parts/glm5.1-storage-part05.juju",19194858195ull,7734,853,0,57,1282371072ull,17908695232ull,1282371072ull,17,4,16,6},
    {6,"parts/glm5.1-storage-part06.juju",23175639236ull,9292,1024,0,76,1709828096ull,21461205184ull,1709828096ull,21,4,20,6},
    {7,"parts/glm5.1-storage-part07.juju",21379413903ull,8527,939,0,76,1709828096ull,19665322176ull,1709828096ull,25,4,24,6},
    {8,"parts/glm5.1-storage-part08.juju",19198789856ull,7734,853,0,57,1282371072ull,17912627392ull,1282371072ull,29,4,28,6},
    {9,"parts/glm5.1-storage-part09.juju",23199229089ull,9292,1024,0,76,1709828096ull,21484798144ull,1709828096ull,33,4,32,6},
    {10,"parts/glm5.1-storage-part10.juju",21380594405ull,8527,939,0,76,1709828096ull,19666501824ull,1709828096ull,37,4,36,6},
    {11,"parts/glm5.1-storage-part11.juju",19207047118ull,7734,853,0,57,1282371072ull,17920884928ull,1282371072ull,41,4,40,6},
    {12,"parts/glm5.1-storage-part12.juju",23185076178ull,9292,1024,0,76,1709828096ull,21470642368ull,1709828096ull,45,4,44,6},
    {13,"parts/glm5.1-storage-part13.juju",21374696840ull,8527,939,0,76,1709828096ull,19660603584ull,1709828096ull,49,4,48,6},
    {14,"parts/glm5.1-storage-part14.juju",19198790663ull,7734,853,0,57,1282371072ull,17912627392ull,1282371072ull,53,4,52,6},
    {15,"parts/glm5.1-storage-part15.juju",23179571825ull,9292,1024,0,76,1709828096ull,21465137344ull,1709828096ull,57,4,56,6},
    {16,"parts/glm5.1-storage-part16.juju",21513876253ull,8527,939,0,76,1709828096ull,19799802048ull,1709828096ull,61,4,60,6},
    {17,"parts/glm5.1-storage-part17.juju",19454346168ull,7734,853,0,57,1282371072ull,18168217792ull,1282371072ull,65,4,64,6},
    {18,"parts/glm5.1-storage-part18.juju",23524768312ull,9292,1024,0,76,1709828096ull,21810380992ull,1709828096ull,69,4,68,6},
    {19,"parts/glm5.1-storage-part19.juju",21713995761ull,8527,939,0,76,1709828096ull,19999948992ull,1709828096ull,73,4,72,6},
    {20,"parts/glm5.1-storage-part20.juju",16166004403ull,6449,597,0,1076,3447287132ull,12715622592ull,3447307712ull,77,3,76,5},
    {21,"parts/glm5.1-storage-part21.juju",41264048735ull,7760,768,256,80,5574924288ull,35685203968ull,5574924288ull,80,5,79,6}
};

uint32_t glm5_storage_part_count(void) {
    return (uint32_t)(sizeof(kGlm5Parts) / sizeof(kGlm5Parts[0]));
}

const glm5_storage_part_spec_t* glm5_storage_part_at(uint32_t index) {
    return index < glm5_storage_part_count() ? &kGlm5Parts[index] : 0;
}

const glm5_storage_part_spec_t* glm5_storage_part_by_id(uint32_t part) {
    return part >= 1 && part <= glm5_storage_part_count() ? &kGlm5Parts[part - 1] : 0;
}
