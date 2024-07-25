#include "daxa/daxa.inl"
#include "hello_triangle_shared.inl"

DAXA_DECL_PUSH_CONSTANT(Push, p)

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 0) out vec4 v_col;
void main()
{
    vec2 pos = vec2(
    float(1 - gl_VertexIndex) * 0.5,
    float((gl_VertexIndex & 1) * 2 - 1) * 0.5
    );

    gl_Position = vec4(pos, 0.0, 1.0);
    v_col = vec4(p.triangle_color, 1.0);
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) in vec4 v_col;
layout(location = 0) out vec4 color;
void main()
{
    color = v_col;
}

#endif