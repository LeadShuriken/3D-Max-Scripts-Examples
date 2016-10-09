 Shader "Philips/RoomFinal" {
    Properties { 
    	_MainTex ("Texture", 2D) = "white" {} 
    	_EmptyRoom ("EmptyRoom", 2D) = "white" {} 
    	_NormalContrast ("NormalContrast", Range(0,1)) = 0.7
    	_Dimming ("Dimming", Range(0,1)) = 1.0
    }
    SubShader {
		Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }
		
		  Name "passone"
		 
		  CGPROGRAM
		  #pragma surface surf SimpleLambert fullforwardshadows noambient
 		  float _Dimming = 1;
		  float _NormalContrast;
		  
		  half4 LightingSimpleLambert (SurfaceOutput s, half3 lightDir, half atten) {
			  half NdotL = dot (s.Normal, lightDir);
			  // Inspired by half Lambert shader
			  half diff = NdotL * _NormalContrast + (1-_NormalContrast);
			  half4 c;
			  c.rgb = s.Albedo * s.Alpha * _LightColor0.rgb * (diff * atten * _Dimming * 2);
			  c.a = s.Alpha;
			  return c;
		  } 

		  struct Input {
			  float2 uv_MainTex;
			  float4 screenPos;
		  };
		  sampler2D _MainTex;
		  sampler2D _EmptyRoom;
                
		  void surf (Input IN, inout SurfaceOutput o) {
		  	 
		  	 
		  	  float4 screenP = IN.screenPos;
		  	  screenP.x += 0/_ScreenParams.y;
		  	  screenP.y -= 4/_ScreenParams.x;
			  float2 screenUV = screenP.xy / IN.screenPos.w;
			  o.Albedo = tex2D (_EmptyRoom, screenUV).rgb;
			  o.Alpha = tex2D (_EmptyRoom, screenUV).a;
			  o.Emission = tex2D (_MainTex, screenUV).rgb;
		  }
		  ENDCG
	  
	 

    } 
    Fallback "Diffuse"
  }