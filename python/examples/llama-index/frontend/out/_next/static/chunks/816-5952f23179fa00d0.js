"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[816],{703:function(e,t,n){n.d(t,{default:function(){return o.a}});var r=n(7447),o=n.n(r)},9079:function(e,t,n){var r,o;e.exports=(null==(r=n.g.process)?void 0:r.env)&&"object"==typeof(null==(o=n.g.process)?void 0:o.env)?n.g.process:n(3127)},1749:function(e,t,n){Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"Image",{enumerable:!0,get:function(){return _}});let r=n(6921),o=n(1884),i=n(3827),a=o._(n(4090)),l=r._(n(9542)),s=r._(n(2251)),u=n(8630),c=n(6906),d=n(337);n(6184);let f=n(6993),p=r._(n(536)),g={deviceSizes:[640,750,828,1080,1200,1920,2048,3840],imageSizes:[16,32,48,64,96,128,256,384],path:"/_next/image",loader:"default",dangerouslyAllowSVG:!1,unoptimized:!0};function m(e,t,n,r,o,i){let a=null==e?void 0:e.src;e&&e["data-loaded-src"]!==a&&(e["data-loaded-src"]=a,("decode"in e?e.decode():Promise.resolve()).catch(()=>{}).then(()=>{if(e.parentElement&&e.isConnected){if("empty"!==t&&o(!0),null==n?void 0:n.current){let t=new Event("load");Object.defineProperty(t,"target",{writable:!1,value:e});let r=!1,o=!1;n.current({...t,nativeEvent:t,currentTarget:e,target:e,isDefaultPrevented:()=>r,isPropagationStopped:()=>o,persist:()=>{},preventDefault:()=>{r=!0,t.preventDefault()},stopPropagation:()=>{o=!0,t.stopPropagation()}})}(null==r?void 0:r.current)&&r.current(e)}}))}function h(e){let[t,n]=a.version.split(".",2),r=parseInt(t,10),o=parseInt(n,10);return r>18||18===r&&o>=3?{fetchPriority:e}:{fetchpriority:e}}let y=(0,a.forwardRef)((e,t)=>{let{src:n,srcSet:r,sizes:o,height:l,width:s,decoding:u,className:c,style:d,fetchPriority:f,placeholder:p,loading:g,unoptimized:y,fill:v,onLoadRef:_,onLoadingCompleteRef:b,setBlurComplete:w,setShowAltText:E,onLoad:x,onError:S,...C}=e;return(0,i.jsx)("img",{...C,...h(f),loading:g,width:s,height:l,decoding:u,"data-nimg":v?"fill":"1",className:c,style:d,sizes:o,srcSet:r,src:n,ref:(0,a.useCallback)(e=>{t&&("function"==typeof t?t(e):"object"==typeof t&&(t.current=e)),e&&(S&&(e.src=e.src),e.complete&&m(e,p,_,b,w,y))},[n,p,_,b,w,S,y,t]),onLoad:e=>{m(e.currentTarget,p,_,b,w,y)},onError:e=>{E(!0),"empty"!==p&&w(!0),S&&S(e)}})});function v(e){let{isAppRouter:t,imgAttributes:n}=e,r={as:"image",imageSrcSet:n.srcSet,imageSizes:n.sizes,crossOrigin:n.crossOrigin,referrerPolicy:n.referrerPolicy,...h(n.fetchPriority)};return t&&l.default.preload?(l.default.preload(n.src,r),null):(0,i.jsx)(s.default,{children:(0,i.jsx)("link",{rel:"preload",href:n.srcSet?void 0:n.src,...r},"__nimg-"+n.src+n.srcSet+n.sizes)})}let _=(0,a.forwardRef)((e,t)=>{let n=(0,a.useContext)(f.RouterContext),r=(0,a.useContext)(d.ImageConfigContext),o=(0,a.useMemo)(()=>{let e=g||r||c.imageConfigDefault,t=[...e.deviceSizes,...e.imageSizes].sort((e,t)=>e-t),n=e.deviceSizes.sort((e,t)=>e-t);return{...e,allSizes:t,deviceSizes:n}},[r]),{onLoad:l,onLoadingComplete:s}=e,m=(0,a.useRef)(l);(0,a.useEffect)(()=>{m.current=l},[l]);let h=(0,a.useRef)(s);(0,a.useEffect)(()=>{h.current=s},[s]);let[_,b]=(0,a.useState)(!1),[w,E]=(0,a.useState)(!1),{props:x,meta:S}=(0,u.getImgProps)(e,{defaultLoader:p.default,imgConf:o,blurComplete:_,showAltText:w});return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(y,{...x,unoptimized:S.unoptimized,placeholder:S.placeholder,fill:S.fill,onLoadRef:m,onLoadingCompleteRef:h,setBlurComplete:b,setShowAltText:E,ref:t}),S.priority?(0,i.jsx)(v,{isAppRouter:!n,imgAttributes:x}):null]})});("function"==typeof t.default||"object"==typeof t.default&&null!==t.default)&&void 0===t.default.__esModule&&(Object.defineProperty(t.default,"__esModule",{value:!0}),Object.assign(t.default,t),e.exports=t.default)},3127:function(e){!function(){var t={229:function(e){var t,n,r,o=e.exports={};function i(){throw Error("setTimeout has not been defined")}function a(){throw Error("clearTimeout has not been defined")}function l(e){if(t===setTimeout)return setTimeout(e,0);if((t===i||!t)&&setTimeout)return t=setTimeout,setTimeout(e,0);try{return t(e,0)}catch(n){try{return t.call(null,e,0)}catch(n){return t.call(this,e,0)}}}!function(){try{t="function"==typeof setTimeout?setTimeout:i}catch(e){t=i}try{n="function"==typeof clearTimeout?clearTimeout:a}catch(e){n=a}}();var s=[],u=!1,c=-1;function d(){u&&r&&(u=!1,r.length?s=r.concat(s):c=-1,s.length&&f())}function f(){if(!u){var e=l(d);u=!0;for(var t=s.length;t;){for(r=s,s=[];++c<t;)r&&r[c].run();c=-1,t=s.length}r=null,u=!1,function(e){if(n===clearTimeout)return clearTimeout(e);if((n===a||!n)&&clearTimeout)return n=clearTimeout,clearTimeout(e);try{n(e)}catch(t){try{return n.call(null,e)}catch(t){return n.call(this,e)}}}(e)}}function p(e,t){this.fun=e,this.array=t}function g(){}o.nextTick=function(e){var t=Array(arguments.length-1);if(arguments.length>1)for(var n=1;n<arguments.length;n++)t[n-1]=arguments[n];s.push(new p(e,t)),1!==s.length||u||l(f)},p.prototype.run=function(){this.fun.apply(null,this.array)},o.title="browser",o.browser=!0,o.env={},o.argv=[],o.version="",o.versions={},o.on=g,o.addListener=g,o.once=g,o.off=g,o.removeListener=g,o.removeAllListeners=g,o.emit=g,o.prependListener=g,o.prependOnceListener=g,o.listeners=function(e){return[]},o.binding=function(e){throw Error("process.binding is not supported")},o.cwd=function(){return"/"},o.chdir=function(e){throw Error("process.chdir is not supported")},o.umask=function(){return 0}}},n={};function r(e){var o=n[e];if(void 0!==o)return o.exports;var i=n[e]={exports:{}},a=!0;try{t[e](i,i.exports,r),a=!1}finally{a&&delete n[e]}return i.exports}r.ab="//";var o=r(229);e.exports=o}()},5827:function(e,t,n){Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"AmpStateContext",{enumerable:!0,get:function(){return r}});let r=n(6921)._(n(4090)).default.createContext({})},3044:function(e,t){function n(e){let{ampFirst:t=!1,hybrid:n=!1,hasQuery:r=!1}=void 0===e?{}:e;return t||n&&r}Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"isInAmpMode",{enumerable:!0,get:function(){return n}})},8630:function(e,t,n){Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"getImgProps",{enumerable:!0,get:function(){return l}}),n(6184);let r=n(7160),o=n(6906);function i(e){return void 0!==e.default}function a(e){return void 0===e?e:"number"==typeof e?Number.isFinite(e)?e:NaN:"string"==typeof e&&/^[0-9]+$/.test(e)?parseInt(e,10):NaN}function l(e,t){var n;let l,s,u,{src:c,sizes:d,unoptimized:f=!1,priority:p=!1,loading:g,className:m,quality:h,width:y,height:v,fill:_=!1,style:b,onLoad:w,onLoadingComplete:E,placeholder:x="empty",blurDataURL:S,fetchPriority:C,layout:j,objectFit:O,objectPosition:T,lazyBoundary:R,lazyRoot:k,...P}=e,{imgConf:A,showAltText:I,blurComplete:M,defaultLoader:D}=t,L=A||o.imageConfigDefault;if("allSizes"in L)l=L;else{let e=[...L.deviceSizes,...L.imageSizes].sort((e,t)=>e-t),t=L.deviceSizes.sort((e,t)=>e-t);l={...L,allSizes:e,deviceSizes:t}}let V=P.loader||D;delete P.loader,delete P.srcSet;let N="__next_img_default"in V;if(N){if("custom"===l.loader)throw Error('Image with src "'+c+'" is missing "loader" prop.\nRead more: https://nextjs.org/docs/messages/next-image-missing-loader')}else{let e=V;V=t=>{let{config:n,...r}=t;return e(r)}}if(j){"fill"===j&&(_=!0);let e={intrinsic:{maxWidth:"100%",height:"auto"},responsive:{width:"100%",height:"auto"}}[j];e&&(b={...b,...e});let t={responsive:"100vw",fill:"100vw"}[j];t&&!d&&(d=t)}let z="",F=a(y),U=a(v);if("object"==typeof(n=c)&&(i(n)||void 0!==n.src)){let e=i(c)?c.default:c;if(!e.src)throw Error("An object should only be passed to the image component src parameter if it comes from a static image import. It must include src. Received "+JSON.stringify(e));if(!e.height||!e.width)throw Error("An object should only be passed to the image component src parameter if it comes from a static image import. It must include height and width. Received "+JSON.stringify(e));if(s=e.blurWidth,u=e.blurHeight,S=S||e.blurDataURL,z=e.src,!_){if(F||U){if(F&&!U){let t=F/e.width;U=Math.round(e.height*t)}else if(!F&&U){let t=U/e.height;F=Math.round(e.width*t)}}else F=e.width,U=e.height}}let W=!p&&("lazy"===g||void 0===g);(!(c="string"==typeof c?c:z)||c.startsWith("data:")||c.startsWith("blob:"))&&(f=!0,W=!1),l.unoptimized&&(f=!0),N&&c.endsWith(".svg")&&!l.dangerouslyAllowSVG&&(f=!0),p&&(C="high");let B=a(h),J=Object.assign(_?{position:"absolute",height:"100%",width:"100%",left:0,top:0,right:0,bottom:0,objectFit:O,objectPosition:T}:{},I?{}:{color:"transparent"},b),G=M||"empty"===x?null:"blur"===x?'url("data:image/svg+xml;charset=utf-8,'+(0,r.getImageBlurSvg)({widthInt:F,heightInt:U,blurWidth:s,blurHeight:u,blurDataURL:S||"",objectFit:J.objectFit})+'")':'url("'+x+'")',q=G?{backgroundSize:J.objectFit||"cover",backgroundPosition:J.objectPosition||"50% 50%",backgroundRepeat:"no-repeat",backgroundImage:G}:{},H=function(e){let{config:t,src:n,unoptimized:r,width:o,quality:i,sizes:a,loader:l}=e;if(r)return{src:n,srcSet:void 0,sizes:void 0};let{widths:s,kind:u}=function(e,t,n){let{deviceSizes:r,allSizes:o}=e;if(n){let e=/(^|\s)(1?\d?\d)vw/g,t=[];for(let r;r=e.exec(n);r)t.push(parseInt(r[2]));if(t.length){let e=.01*Math.min(...t);return{widths:o.filter(t=>t>=r[0]*e),kind:"w"}}return{widths:o,kind:"w"}}return"number"!=typeof t?{widths:r,kind:"w"}:{widths:[...new Set([t,2*t].map(e=>o.find(t=>t>=e)||o[o.length-1]))],kind:"x"}}(t,o,a),c=s.length-1;return{sizes:a||"w"!==u?a:"100vw",srcSet:s.map((e,r)=>l({config:t,src:n,quality:i,width:e})+" "+("w"===u?e:r+1)+u).join(", "),src:l({config:t,src:n,quality:i,width:s[c]})}}({config:l,src:c,unoptimized:f,width:F,quality:B,sizes:d,loader:V});return{props:{...P,loading:W?"lazy":g,fetchPriority:C,width:F,height:U,decoding:"async",className:m,style:{...J,...q},sizes:H.sizes,srcSet:H.srcSet,src:H.src},meta:{unoptimized:f,priority:p,placeholder:x,fill:_}}}},2251:function(e,t,n){Object.defineProperty(t,"__esModule",{value:!0}),function(e,t){for(var n in t)Object.defineProperty(e,n,{enumerable:!0,get:t[n]})}(t,{defaultHead:function(){return d},default:function(){return m}});let r=n(6921),o=n(1884),i=n(3827),a=o._(n(4090)),l=r._(n(7392)),s=n(5827),u=n(7484),c=n(3044);function d(e){void 0===e&&(e=!1);let t=[(0,i.jsx)("meta",{charSet:"utf-8"})];return e||t.push((0,i.jsx)("meta",{name:"viewport",content:"width=device-width"})),t}function f(e,t){return"string"==typeof t||"number"==typeof t?e:t.type===a.default.Fragment?e.concat(a.default.Children.toArray(t.props.children).reduce((e,t)=>"string"==typeof t||"number"==typeof t?e:e.concat(t),[])):e.concat(t)}n(6184);let p=["name","httpEquiv","charSet","itemProp"];function g(e,t){let{inAmpMode:n}=t;return e.reduce(f,[]).reverse().concat(d(n).reverse()).filter(function(){let e=new Set,t=new Set,n=new Set,r={};return o=>{let i=!0,a=!1;if(o.key&&"number"!=typeof o.key&&o.key.indexOf("$")>0){a=!0;let t=o.key.slice(o.key.indexOf("$")+1);e.has(t)?i=!1:e.add(t)}switch(o.type){case"title":case"base":t.has(o.type)?i=!1:t.add(o.type);break;case"meta":for(let e=0,t=p.length;e<t;e++){let t=p[e];if(o.props.hasOwnProperty(t)){if("charSet"===t)n.has(t)?i=!1:n.add(t);else{let e=o.props[t],n=r[t]||new Set;("name"!==t||!a)&&n.has(e)?i=!1:(n.add(e),r[t]=n)}}}}return i}}()).reverse().map((e,t)=>{let r=e.key||t;if(!n&&"link"===e.type&&e.props.href&&["https://fonts.googleapis.com/css","https://use.typekit.net/"].some(t=>e.props.href.startsWith(t))){let t={...e.props||{}};return t["data-href"]=t.href,t.href=void 0,t["data-optimized-fonts"]=!0,a.default.cloneElement(e,t)}return a.default.cloneElement(e,{key:r})})}let m=function(e){let{children:t}=e,n=(0,a.useContext)(s.AmpStateContext),r=(0,a.useContext)(u.HeadManagerContext);return(0,i.jsx)(l.default,{reduceComponentsToState:g,headManager:r,inAmpMode:(0,c.isInAmpMode)(n),children:t})};("function"==typeof t.default||"object"==typeof t.default&&null!==t.default)&&void 0===t.default.__esModule&&(Object.defineProperty(t.default,"__esModule",{value:!0}),Object.assign(t.default,t),e.exports=t.default)},7160:function(e,t){function n(e){let{widthInt:t,heightInt:n,blurWidth:r,blurHeight:o,blurDataURL:i,objectFit:a}=e,l=r?40*r:t,s=o?40*o:n,u=l&&s?"viewBox='0 0 "+l+" "+s+"'":"";return"%3Csvg xmlns='http://www.w3.org/2000/svg' "+u+"%3E%3Cfilter id='b' color-interpolation-filters='sRGB'%3E%3CfeGaussianBlur stdDeviation='20'/%3E%3CfeColorMatrix values='1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 100 -1' result='s'/%3E%3CfeFlood x='0' y='0' width='100%25' height='100%25'/%3E%3CfeComposite operator='out' in='s'/%3E%3CfeComposite in2='SourceGraphic'/%3E%3CfeGaussianBlur stdDeviation='20'/%3E%3C/filter%3E%3Cimage width='100%25' height='100%25' x='0' y='0' preserveAspectRatio='"+(u?"none":"contain"===a?"xMidYMid":"cover"===a?"xMidYMid slice":"none")+"' style='filter: url(%23b);' href='"+i+"'/%3E%3C/svg%3E"}Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"getImageBlurSvg",{enumerable:!0,get:function(){return n}})},337:function(e,t,n){Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"ImageConfigContext",{enumerable:!0,get:function(){return i}});let r=n(6921)._(n(4090)),o=n(6906),i=r.default.createContext(o.imageConfigDefault)},6906:function(e,t){Object.defineProperty(t,"__esModule",{value:!0}),function(e,t){for(var n in t)Object.defineProperty(e,n,{enumerable:!0,get:t[n]})}(t,{VALID_LOADERS:function(){return n},imageConfigDefault:function(){return r}});let n=["default","imgix","cloudinary","akamai","custom"],r={deviceSizes:[640,750,828,1080,1200,1920,2048,3840],imageSizes:[16,32,48,64,96,128,256,384],path:"/_next/image",loader:"default",loaderFile:"",domains:[],disableStaticImages:!1,minimumCacheTTL:60,formats:["image/webp"],dangerouslyAllowSVG:!1,contentSecurityPolicy:"script-src 'none'; frame-src 'none'; sandbox;",contentDispositionType:"inline",remotePatterns:[],unoptimized:!1}},7447:function(e,t,n){Object.defineProperty(t,"__esModule",{value:!0}),function(e,t){for(var n in t)Object.defineProperty(e,n,{enumerable:!0,get:t[n]})}(t,{getImageProps:function(){return l},default:function(){return s}});let r=n(6921),o=n(8630),i=n(1749),a=r._(n(536)),l=e=>{let{props:t}=(0,o.getImgProps)(e,{defaultLoader:a.default,imgConf:{deviceSizes:[640,750,828,1080,1200,1920,2048,3840],imageSizes:[16,32,48,64,96,128,256,384],path:"/_next/image",loader:"default",dangerouslyAllowSVG:!1,unoptimized:!0}});for(let[e,n]of Object.entries(t))void 0===n&&delete t[e];return{props:t}},s=i.Image},536:function(e,t){function n(e){let{config:t,src:n,width:r,quality:o}=e;return t.path+"?url="+encodeURIComponent(n)+"&w="+r+"&q="+(o||75)}Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"default",{enumerable:!0,get:function(){return r}}),n.__next_img_default=!0;let r=n},6993:function(e,t,n){Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"RouterContext",{enumerable:!0,get:function(){return r}});let r=n(6921)._(n(4090)).default.createContext(null)},7392:function(e,t,n){Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"default",{enumerable:!0,get:function(){return a}});let r=n(4090),o=r.useLayoutEffect,i=r.useEffect;function a(e){let{headManager:t,reduceComponentsToState:n}=e;function a(){if(t&&t.mountedInstances){let o=r.Children.toArray(Array.from(t.mountedInstances).filter(Boolean));t.updateHead(n(o,e))}}return o(()=>{var n;return null==t||null==(n=t.mountedInstances)||n.add(e.children),()=>{var n;null==t||null==(n=t.mountedInstances)||n.delete(e.children)}}),o(()=>(t&&(t._pendingUpdate=a),()=>{t&&(t._pendingUpdate=a)})),i(()=>(t&&t._pendingUpdate&&(t._pendingUpdate(),t._pendingUpdate=null),()=>{t&&t._pendingUpdate&&(t._pendingUpdate(),t._pendingUpdate=null)})),null}},699:function(e,t,n){/**
 * @license React
 * use-sync-external-store-shim.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var r=n(4090),o="function"==typeof Object.is?Object.is:function(e,t){return e===t&&(0!==e||1/e==1/t)||e!=e&&t!=t},i=r.useState,a=r.useEffect,l=r.useLayoutEffect,s=r.useDebugValue;function u(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!o(e,n)}catch(e){return!0}}var c=void 0===window.document||void 0===window.document.createElement?function(e,t){return t()}:function(e,t){var n=t(),r=i({inst:{value:n,getSnapshot:t}}),o=r[0].inst,c=r[1];return l(function(){o.value=n,o.getSnapshot=t,u(o)&&c({inst:o})},[e,n,t]),a(function(){return u(o)&&c({inst:o}),e(function(){u(o)&&c({inst:o})})},[e]),s(n),n};t.useSyncExternalStore=void 0!==r.useSyncExternalStore?r.useSyncExternalStore:c},2362:function(e,t,n){e.exports=n(699)},5237:function(e,t,n){n.d(t,{RJ:function(){return ey}});var r,o=n(4090),i=n(2362);let a=()=>{},l=a(),s=Object,u=e=>e===l,c=e=>"function"==typeof e,d=(e,t)=>({...e,...t}),f=e=>c(e.then),p=new WeakMap,g=0,m=e=>{let t,n;let r=typeof e,o=e&&e.constructor,i=o==Date;if(s(e)!==e||i||o==RegExp)t=i?e.toJSON():"symbol"==r?e.toString():"string"==r?JSON.stringify(e):""+e;else{if(t=p.get(e))return t;if(t=++g+"~",p.set(e,t),o==Array){for(n=0,t="@";n<e.length;n++)t+=m(e[n])+",";p.set(e,t)}if(o==s){t="#";let r=s.keys(e).sort();for(;!u(n=r.pop());)u(e[n])||(t+=n+":"+m(e[n])+",");p.set(e,t)}}return t},h=new WeakMap,y={},v={},_="undefined",b=typeof document!=_,w=()=>typeof window.requestAnimationFrame!=_,E=(e,t)=>{let n=h.get(e);return[()=>!u(t)&&e.get(t)||y,r=>{if(!u(t)){let o=e.get(t);t in v||(v[t]=o),n[5](t,d(o,r),o||y)}},n[6],()=>!u(t)&&t in v?v[t]:!u(t)&&e.get(t)||y]},x=!0,[S,C]=window.addEventListener?[window.addEventListener.bind(window),window.removeEventListener.bind(window)]:[a,a],j={initFocus:e=>(b&&document.addEventListener("visibilitychange",e),S("focus",e),()=>{b&&document.removeEventListener("visibilitychange",e),C("focus",e)}),initReconnect:e=>{let t=()=>{x=!0,e()},n=()=>{x=!1};return S("online",t),S("offline",n),()=>{C("online",t),C("offline",n)}}},O=!o.useId,T="Deno"in window,R=e=>w()?window.requestAnimationFrame(e):setTimeout(e,1),k=T?o.useEffect:o.useLayoutEffect,P="undefined"!=typeof navigator&&navigator.connection,A=!T&&P&&(["slow-2g","2g"].includes(P.effectiveType)||P.saveData),I=e=>{if(c(e))try{e=e()}catch(t){e=""}let t=e;return[e="string"==typeof e?e:(Array.isArray(e)?e.length:e)?m(e):"",t]},M=0,D=()=>++M;var L={ERROR_REVALIDATE_EVENT:3,FOCUS_EVENT:0,MUTATE_EVENT:2,RECONNECT_EVENT:1};async function V(){for(var e=arguments.length,t=Array(e),n=0;n<e;n++)t[n]=arguments[n];let[r,o,i,a]=t,s=d({populateCache:!0,throwOnError:!0},"boolean"==typeof a?{revalidate:a}:a||{}),p=s.populateCache,g=s.rollbackOnError,m=s.optimisticData,y=!1!==s.revalidate,v=e=>"function"==typeof g?g(e):!1!==g,_=s.throwOnError;if(c(o)){let e=[];for(let t of r.keys())!/^\$(inf|sub)\$/.test(t)&&o(r.get(t)._k)&&e.push(t);return Promise.all(e.map(b))}return b(o);async function b(e){let n;let[o]=I(e);if(!o)return;let[a,s]=E(r,o),[d,g,b,w]=h.get(r),x=d[o],S=()=>y&&(delete b[o],delete w[o],x&&x[0])?x[0](2).then(()=>a().data):a().data;if(t.length<3)return S();let C=i,j=D();g[o]=[j,0];let O=!u(m),T=a(),R=T.data,k=T._c,P=u(k)?R:k;if(O&&s({data:m=c(m)?m(P,R):m,_c:P}),c(C))try{C=C(P)}catch(e){n=e}if(C&&f(C)){if(C=await C.catch(e=>{n=e}),j!==g[o][0]){if(n)throw n;return C}n&&O&&v(n)&&(p=!0,s({data:C=P,_c:l}))}p&&!n&&(c(p)&&(C=p(C,P)),s({data:C,error:l,_c:l})),g[o][1]=D();let A=await S();if(s({_c:l}),n){if(_)throw n;return}return p?A:C}}let N=(e,t)=>{for(let n in e)e[n][0]&&e[n][0](t)},z=(e,t)=>{if(!h.has(e)){let n=d(j,t),r={},o=V.bind(l,e),i=a,s={},u=(e,t)=>{let n=s[e]||[];return s[e]=n,n.push(t),()=>n.splice(n.indexOf(t),1)},c=(t,n,r)=>{e.set(t,n);let o=s[t];if(o)for(let e of o)e(n,r)},f=()=>{if(!h.has(e)&&(h.set(e,[r,{},{},{},o,c,u]),!T)){let t=n.initFocus(setTimeout.bind(l,N.bind(l,r,0))),o=n.initReconnect(setTimeout.bind(l,N.bind(l,r,1)));i=()=>{t&&t(),o&&o(),h.delete(e)}}};return f(),[e,o,f,i]}return[e,h.get(e)[4]]},[F,U]=z(new Map),W=d({onLoadingSlow:a,onSuccess:a,onError:a,onErrorRetry:(e,t,n,r,o)=>{let i=n.errorRetryCount,a=o.retryCount,l=~~((Math.random()+.5)*(1<<(a<8?a:8)))*n.errorRetryInterval;(u(i)||!(a>i))&&setTimeout(r,l,o)},onDiscarded:a,revalidateOnFocus:!0,revalidateOnReconnect:!0,revalidateIfStale:!0,shouldRetryOnError:!0,errorRetryInterval:A?1e4:5e3,focusThrottleInterval:5e3,dedupingInterval:2e3,loadingTimeout:A?5e3:3e3,compare:(e,t)=>m(e)==m(t),isPaused:()=>!1,cache:F,mutate:U,fallback:{}},{isOnline:()=>x,isVisible:()=>{let e=b&&document.visibilityState;return u(e)||"hidden"!==e}}),B=(e,t)=>{let n=d(e,t);if(t){let{use:r,fallback:o}=e,{use:i,fallback:a}=t;r&&i&&(n.use=r.concat(i)),o&&a&&(n.fallback=d(o,a))}return n},J=(0,o.createContext)({}),G=window.__SWR_DEVTOOLS_USE__,q=G?window.__SWR_DEVTOOLS_USE__:[],H=e=>c(e[1])?[e[0],e[1],e[2]||{}]:[e[0],null,(null===e[1]?e[2]:e[1])||{}],$=()=>d(W,(0,o.useContext)(J)),Y=q.concat(e=>(t,n,r)=>{let o=n&&function(){for(var e=arguments.length,r=Array(e),o=0;o<e;o++)r[o]=arguments[o];let[i]=I(t),[,,,a]=h.get(F),l=a[i];return u(l)?n(...r):(delete a[i],l)};return e(t,o,r)}),X=(e,t,n)=>{let r=t[e]||(t[e]=[]);return r.push(n),()=>{let e=r.indexOf(n);e>=0&&(r[e]=r[r.length-1],r.pop())}};G&&(window.__SWR_DEVTOOLS_REACT__=o);let K=o.use||(e=>{if("pending"===e.status)throw e;if("fulfilled"===e.status)return e.value;if("rejected"===e.status)throw e.reason;throw e.status="pending",e.then(t=>{e.status="fulfilled",e.value=t},t=>{e.status="rejected",e.reason=t}),e}),Q={dedupe:!0};s.defineProperty(e=>{let{value:t}=e,n=(0,o.useContext)(J),r=c(t),i=(0,o.useMemo)(()=>r?t(n):t,[r,n,t]),a=(0,o.useMemo)(()=>r?i:B(n,i),[r,n,i]),s=i&&i.provider,u=(0,o.useRef)(l);s&&!u.current&&(u.current=z(s(a.cache||F),i));let f=u.current;return f&&(a.cache=f[0],a.mutate=f[1]),k(()=>{if(f)return f[2]&&f[2](),f[3]},[]),(0,o.createElement)(J.Provider,d(e,{value:a}))},"defaultValue",{value:W});let Z=(r=(e,t,n)=>{let{cache:r,compare:a,suspense:s,fallbackData:f,revalidateOnMount:p,revalidateIfStale:g,refreshInterval:m,refreshWhenHidden:y,refreshWhenOffline:v,keepPreviousData:_}=n,[b,w,x,S]=h.get(r),[C,j]=I(e),P=(0,o.useRef)(!1),A=(0,o.useRef)(!1),M=(0,o.useRef)(C),N=(0,o.useRef)(t),z=(0,o.useRef)(n),F=()=>z.current,U=()=>F().isVisible()&&F().isOnline(),[W,B,J,G]=E(r,C),q=(0,o.useRef)({}).current,H=u(f)?n.fallback[C]:f,$=(e,t)=>{for(let n in q)if("data"===n){if(!a(e[n],t[n])&&(!u(e[n])||!a(ea,t[n])))return!1}else if(t[n]!==e[n])return!1;return!0},Y=(0,o.useMemo)(()=>{let e=!!C&&!!t&&(u(p)?!F().isPaused()&&!s&&(!!u(g)||g):p),n=t=>{let n=d(t);return(delete n._k,e)?{isValidating:!0,isLoading:!0,...n}:n},r=W(),o=G(),i=n(r),a=r===o?i:n(o),l=i;return[()=>{let e=n(W());return $(e,l)?(l.data=e.data,l.isLoading=e.isLoading,l.isValidating=e.isValidating,l.error=e.error,l):(l=e,e)},()=>a]},[r,C]),Z=(0,i.useSyncExternalStore)((0,o.useCallback)(e=>J(C,(t,n)=>{$(n,t)||e()}),[r,C]),Y[0],Y[1]),ee=!P.current,et=b[C]&&b[C].length>0,en=Z.data,er=u(en)?H:en,eo=Z.error,ei=(0,o.useRef)(er),ea=_?u(en)?ei.current:en:er,el=(!et||!!u(eo))&&(ee&&!u(p)?p:!F().isPaused()&&(s?!u(er)&&g:u(er)||g)),es=!!(C&&t&&ee&&el),eu=u(Z.isValidating)?es:Z.isValidating,ec=u(Z.isLoading)?es:Z.isLoading,ed=(0,o.useCallback)(async e=>{let t,r;let o=N.current;if(!C||!o||A.current||F().isPaused())return!1;let i=!0,s=e||{},d=!x[C]||!s.dedupe,f=()=>O?!A.current&&C===M.current&&P.current:C===M.current,p={isValidating:!1,isLoading:!1},g=()=>{B(p)},m=()=>{let e=x[C];e&&e[1]===r&&delete x[C]},h={isValidating:!0};u(W().data)&&(h.isLoading=!0);try{if(d&&(B(h),n.loadingTimeout&&u(W().data)&&setTimeout(()=>{i&&f()&&F().onLoadingSlow(C,n)},n.loadingTimeout),x[C]=[o(j),D()]),[t,r]=x[C],t=await t,d&&setTimeout(m,n.dedupingInterval),!x[C]||x[C][1]!==r)return d&&f()&&F().onDiscarded(C),!1;p.error=l;let e=w[C];if(!u(e)&&(r<=e[0]||r<=e[1]||0===e[1]))return g(),d&&f()&&F().onDiscarded(C),!1;let s=W().data;p.data=a(s,t)?s:t,d&&f()&&F().onSuccess(t,C,n)}catch(n){m();let e=F(),{shouldRetryOnError:t}=e;!e.isPaused()&&(p.error=n,d&&f()&&(e.onError(n,C,e),(!0===t||c(t)&&t(n))&&U()&&e.onErrorRetry(n,C,e,e=>{let t=b[C];t&&t[0]&&t[0](L.ERROR_REVALIDATE_EVENT,e)},{retryCount:(s.retryCount||0)+1,dedupe:!0})))}return i=!1,g(),!0},[C,r]),ef=(0,o.useCallback)(function(){for(var e=arguments.length,t=Array(e),n=0;n<e;n++)t[n]=arguments[n];return V(r,M.current,...t)},[]);if(k(()=>{N.current=t,z.current=n,u(en)||(ei.current=en)}),k(()=>{if(!C)return;let e=ed.bind(l,Q),t=0,n=X(C,b,function(n){let r=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{};if(n==L.FOCUS_EVENT){let n=Date.now();F().revalidateOnFocus&&n>t&&U()&&(t=n+F().focusThrottleInterval,e())}else if(n==L.RECONNECT_EVENT)F().revalidateOnReconnect&&U()&&e();else if(n==L.MUTATE_EVENT)return ed();else if(n==L.ERROR_REVALIDATE_EVENT)return ed(r)});return A.current=!1,M.current=C,P.current=!0,B({_k:j}),el&&(u(er)||T?e():R(e)),()=>{A.current=!0,n()}},[C]),k(()=>{let e;function t(){let t=c(m)?m(W().data):m;t&&-1!==e&&(e=setTimeout(n,t))}function n(){!W().error&&(y||F().isVisible())&&(v||F().isOnline())?ed(Q).then(t):t()}return t(),()=>{e&&(clearTimeout(e),e=-1)}},[m,y,v,C]),(0,o.useDebugValue)(ea),s&&u(er)&&C){if(!O&&T)throw Error("Fallback data is required when using suspense in SSR.");N.current=t,z.current=n,A.current=!1;let e=S[C];if(u(e)||K(ef(e)),u(eo)){let e=ed(Q);u(ea)||(e.status="fulfilled",e.value=!0),K(e)}else throw eo}return{mutate:ef,get data(){return q.data=!0,ea},get error(){return q.error=!0,eo},get isValidating(){return q.isValidating=!0,eu},get isLoading(){return q.isLoading=!0,ec}}},function(){for(var e=arguments.length,t=Array(e),n=0;n<e;n++)t[n]=arguments[n];let o=$(),[i,a,l]=H(t),s=B(o,l),u=r,{use:c}=s,d=(c||[]).concat(Y);for(let e=d.length;e--;)u=d[e](u);return u(i,a||s.fetcher||null,s)});var ee={code:"0",name:"text",parse:e=>{if("string"!=typeof e)throw Error('"text" parts expect a string value.');return{type:"text",value:e}}},et={code:"1",name:"function_call",parse:e=>{if(null==e||"object"!=typeof e||!("function_call"in e)||"object"!=typeof e.function_call||null==e.function_call||!("name"in e.function_call)||!("arguments"in e.function_call)||"string"!=typeof e.function_call.name||"string"!=typeof e.function_call.arguments)throw Error('"function_call" parts expect an object with a "function_call" property.');return{type:"function_call",value:e}}},en={code:"2",name:"data",parse:e=>{if(!Array.isArray(e))throw Error('"data" parts expect an array value.');return{type:"data",value:e}}},er={code:"3",name:"error",parse:e=>{if("string"!=typeof e)throw Error('"error" parts expect a string value.');return{type:"error",value:e}}},eo={code:"4",name:"assistant_message",parse:e=>{if(null==e||"object"!=typeof e||!("id"in e)||!("role"in e)||!("content"in e)||"string"!=typeof e.id||"string"!=typeof e.role||"assistant"!==e.role||!Array.isArray(e.content)||!e.content.every(e=>null!=e&&"object"==typeof e&&"type"in e&&"text"===e.type&&"text"in e&&null!=e.text&&"object"==typeof e.text&&"value"in e.text&&"string"==typeof e.text.value))throw Error('"assistant_message" parts expect an object with an "id", "role", and "content" property.');return{type:"assistant_message",value:e}}},ei={code:"5",name:"assistant_control_data",parse:e=>{if(null==e||"object"!=typeof e||!("threadId"in e)||!("messageId"in e)||"string"!=typeof e.threadId||"string"!=typeof e.messageId)throw Error('"assistant_control_data" parts expect an object with a "threadId" and "messageId" property.');return{type:"assistant_control_data",value:{threadId:e.threadId,messageId:e.messageId}}}},ea={code:"6",name:"data_message",parse:e=>{if(null==e||"object"!=typeof e||!("role"in e)||!("data"in e)||"string"!=typeof e.role||"data"!==e.role)throw Error('"data_message" parts expect an object with a "role" and "data" property.');return{type:"data_message",value:e}}},el={code:"7",name:"tool_calls",parse:e=>{if(null==e||"object"!=typeof e||!("tool_calls"in e)||"object"!=typeof e.tool_calls||null==e.tool_calls||!Array.isArray(e.tool_calls)||e.tool_calls.some(e=>{null!=e&&"object"==typeof e&&"id"in e&&"string"==typeof e.id&&"type"in e&&"string"==typeof e.type&&"function"in e&&null!=e.function&&"object"==typeof e.function&&"arguments"in e.function&&"string"==typeof e.function.name&&e.function.arguments}))throw Error('"tool_calls" parts expect an object with a ToolCallPayload.');return{type:"tool_calls",value:e}}},es={[ee.code]:ee,[et.code]:et,[en.code]:en,[er.code]:er,[eo.code]:eo,[ei.code]:ei,[ea.code]:ea,[el.code]:el};ee.name,ee.code,et.name,et.code,en.name,en.code,er.name,er.code,eo.name,eo.code,ei.name,ei.code,ea.name,ea.code,el.name,el.code;var eu=[ee,et,en,er,eo,ei,ea,el].map(e=>e.code),ec=e=>{let t=e.indexOf(":");if(-1===t)throw Error("Failed to parse stream string. No separator found.");let n=e.slice(0,t);if(!eu.includes(n))throw Error("Failed to parse stream string. Invalid code ".concat(n,"."));let r=JSON.parse(e.slice(t+1));return es[n].parse(r)};async function*ed(e){let{isAborted:t}=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{},n=new TextDecoder,r=[],o=0;for(;;){let{value:i}=await e.read();if(i&&(r.push(i),o+=i.length,10!==i[i.length-1]))continue;if(0===r.length)break;let a=function(e,t){let n=new Uint8Array(t),r=0;for(let t of e)n.set(t,r),r+=t.length;return e.length=0,n}(r,o);for(let e of(o=0,n.decode(a,{stream:!0}).split("\n").filter(e=>""!==e).map(ec)))yield e;if(null==t?void 0:t()){e.cancel();break}}}var ef=function(e){let t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:21;return function(){let n=arguments.length>0&&void 0!==arguments[0]?arguments[0]:t,r="",o=n;for(;o--;)r+=e[Math.random()*e.length|0];return r}}("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",7);async function ep(e){let{reader:t,abortControllerRef:n,update:r,onFinish:o,generateId:i=ef,getCurrentDate:a=()=>new Date}=e,l=a(),s={data:[]};for await(let{type:e,value:o}of ed(t,{isAborted:()=>(null==n?void 0:n.current)===null})){"text"===e&&(s.text?s.text={...s.text,content:(s.text.content||"")+o}:s.text={id:i(),role:"assistant",content:o,createdAt:l});let t=null;"function_call"===e&&(s.function_call={id:i(),role:"assistant",content:"",function_call:o.function_call,name:o.function_call.name,createdAt:l},t=s.function_call);let n=null;"tool_calls"===e&&(s.tool_calls={id:i(),role:"assistant",content:"",tool_calls:o.tool_calls,createdAt:l},n=s.tool_calls),"data"===e&&s.data.push(...o),r([t,n,s.text].filter(Boolean),[...s.data])}return null==o||o(s),{messages:[s.text,s.function_call,s.tool_calls].filter(Boolean),data:s.data}}async function eg(e){var t;let{api:n,messages:r,body:o,credentials:i,headers:a,abortController:l,appendMessage:s,restoreMessagesOnFailure:u,onResponse:c,onUpdate:d,onFinish:f,generateId:p}=e,g=await fetch(n,{method:"POST",body:JSON.stringify({messages:r,...o}),headers:{"Content-Type":"application/json",...a},signal:null==(t=null==l?void 0:l())?void 0:t.signal,credentials:i}).catch(e=>{throw u(),e});if(c)try{await c(g)}catch(e){throw e}if(!g.ok)throw u(),Error(await g.text()||"Failed to fetch the chat response.");if(!g.body)throw Error("The response body is empty.");let m=g.body.getReader();if("true"===g.headers.get("X-Experimental-Stream-Data"))return await ep({reader:m,abortControllerRef:null!=l?{current:l()}:void 0,update:d,onFinish(e){f&&null!=e.text&&f(e.text)},generateId:p});{let e=new Date,t=function(e){let t=new TextDecoder;return e?function(e){return t.decode(e,{stream:!0}).split("\n").filter(e=>""!==e).map(ec).filter(Boolean)}:function(e){return e?t.decode(e,{stream:!0}):""}}(!1),n="",r={id:p(),createdAt:e,content:"",role:"assistant"};for(;;){let{done:e,value:o}=await m.read();if(e)break;if((n+=t(o)).startsWith('{"function_call":')?r.function_call=n:n.startsWith('{"tool_calls":')?r.tool_calls=n:r.content=n,s({...r}),(null==l?void 0:l())===null){m.cancel();break}}if(n.startsWith('{"function_call":')){let e=JSON.parse(n).function_call;r.function_call=e,s({...r})}if(n.startsWith('{"tool_calls":')){let e=JSON.parse(n).tool_calls;r.tool_calls=e,s({...r})}return f&&f(r),r}}async function em(e){let{getStreamedResponse:t,experimental_onFunctionCall:n,experimental_onToolCall:r,updateChatRequest:o,getCurrentMessages:i}=e;for(;;){let e=await t();if("messages"in e){let t=!1;for(let a of e.messages)if(void 0!==a.function_call&&"string"!=typeof a.function_call||void 0!==a.tool_calls&&"string"!=typeof a.tool_calls){if(t=!0,n){let e=a.function_call;if("object"!=typeof e){console.warn("experimental_onFunctionCall should not be defined when using tools");continue}let r=await n(i(),e);if(void 0===r){t=!1;break}o(r)}if(r){let e=a.tool_calls;if(!Array.isArray(e)||e.some(e=>"object"!=typeof e)){console.warn("experimental_onToolCall should not be defined when using tools");continue}let n=await r(i(),e);if(void 0===n){t=!1;break}o(n)}}if(!t)break}else{let t=function(e){for(let t of e.messages){if(void 0!==t.tool_calls)for(let e of t.tool_calls)"object"==typeof e&&e.function.arguments&&"string"!=typeof e.function.arguments&&(e.function.arguments=JSON.stringify(e.function.arguments));void 0!==t.function_call&&"object"==typeof t.function_call&&t.function_call.arguments&&"string"!=typeof t.function_call.arguments&&(t.function_call.arguments=JSON.stringify(t.function_call.arguments))}};if((void 0===e.function_call||"string"==typeof e.function_call)&&(void 0===e.tool_calls||"string"==typeof e.tool_calls))break;if(n){let r=e.function_call;if("object"!=typeof r){console.warn("experimental_onFunctionCall should not be defined when using tools");continue}let a=await n(i(),r);if(void 0===a)break;t(a),o(a)}if(r){let n=e.tool_calls;if("object"!=typeof n){console.warn("experimental_onToolCall should not be defined when using functions");continue}let a=await r(i(),n);if(void 0===a)break;t(a),o(a)}}}}var eh=async(e,t,n,r,o,i,a,l,s,u,c,d)=>{var f,p;let g=a.current;n(t.messages,!1);let m=d?t.messages:t.messages.map(e=>{let{role:t,content:n,name:r,function_call:o,tool_calls:i,tool_call_id:a}=e;return{role:t,content:n,tool_call_id:a,...void 0!==r&&{name:r},...void 0!==o&&{function_call:o},...void 0!==i&&{tool_calls:i}}});if("string"!=typeof e){let r={id:s(),createdAt:new Date,content:"",role:"assistant"};async function h(e){let{content:o,ui:i,next:a}=await e;r.content=o,r.ui=await i,n([...t.messages,{...r}],!1),a&&await h(a)}try{let n=e({messages:m,data:t.data});await h(n)}catch(e){throw n(g,!1),e}return u&&u(r),r}return await eg({api:e,messages:m,body:{data:t.data,...i.current.body,...null==(f=t.options)?void 0:f.body,...void 0!==t.functions&&{functions:t.functions},...void 0!==t.function_call&&{function_call:t.function_call},...void 0!==t.tools&&{tools:t.tools},...void 0!==t.tool_choice&&{tool_choice:t.tool_choice}},credentials:i.current.credentials,headers:{...i.current.headers,...null==(p=t.options)?void 0:p.headers},abortController:()=>l.current,appendMessage(e){n([...t.messages,e],!1)},restoreMessagesOnFailure(){n(g,!1)},onResponse:c,onUpdate(e,i){n([...t.messages,...e],!1),r([...o||[],...i||[]],!1)},onFinish:u,generateId:s})};function ey(){let{api:e="/api/chat",id:t,initialMessages:n,initialInput:r="",sendExtraMessageFields:i,experimental_onFunctionCall:a,experimental_onToolCall:l,onResponse:s,onFinish:u,onError:c,credentials:d,headers:f,body:p,generateId:g=ef}=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},m=(0,o.useId)(),h=null!=t?t:m,y="string"==typeof e?[e,h]:h,[v]=(0,o.useState)([]),{data:_,mutate:b}=Z([y,"messages"],null,{fallbackData:null!=n?n:v}),{data:w=!1,mutate:E}=Z([y,"loading"],null),{data:x,mutate:S}=Z([y,"streamData"],null),{data:C,mutate:j}=Z([y,"error"],null),O=(0,o.useRef)(_||[]);(0,o.useEffect)(()=>{O.current=_||[]},[_]);let T=(0,o.useRef)(null),R=(0,o.useRef)({credentials:d,headers:f,body:p});(0,o.useEffect)(()=>{R.current={credentials:d,headers:f,body:p}},[d,f,p]);let k=(0,o.useCallback)(async t=>{try{E(!0),j(void 0);let n=new AbortController;T.current=n,await em({getStreamedResponse:()=>eh(e,t,b,S,x,R,O,T,g,u,s,i),experimental_onFunctionCall:a,experimental_onToolCall:l,updateChatRequest:e=>{t=e},getCurrentMessages:()=>O.current}),T.current=null}catch(e){if("AbortError"===e.name)return T.current=null,null;c&&e instanceof Error&&c(e),j(e)}finally{E(!1)}},[b,E,e,R,s,u,c,j,S,x,i,a,l,O,T,g]),P=(0,o.useCallback)(async function(e){let{options:t,functions:n,function_call:r,tools:o,tool_choice:i,data:a}=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{};return e.id||(e.id=g()),k({messages:O.current.concat(e),options:t,data:a,...void 0!==n&&{functions:n},...void 0!==r&&{function_call:r},...void 0!==o&&{tools:o},...void 0!==i&&{tool_choice:i}})},[k,g]),A=(0,o.useCallback)(async function(){let{options:e,functions:t,function_call:n,tools:r,tool_choice:o}=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{};return 0===O.current.length?null:"assistant"===O.current[O.current.length-1].role?k({messages:O.current.slice(0,-1),options:e,...void 0!==t&&{functions:t},...void 0!==n&&{function_call:n},...void 0!==r&&{tools:r},...void 0!==o&&{tool_choice:o}}):k({messages:O.current,options:e,...void 0!==t&&{functions:t},...void 0!==n&&{function_call:n},...void 0!==r&&{tools:r},...void 0!==o&&{tool_choice:o}})},[k]),I=(0,o.useCallback)(()=>{T.current&&(T.current.abort(),T.current=null)},[]),M=(0,o.useCallback)(e=>{b(e,!1),O.current=e},[b]),[D,L]=(0,o.useState)(r),V=(0,o.useCallback)(function(e){let t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{},n=arguments.length>2?arguments[2]:void 0;n&&(R.current={...R.current,...n}),e.preventDefault(),D&&(P({content:D,role:"user",createdAt:new Date},t),L(""))},[D,P]);return{messages:_||[],error:C,append:P,reload:A,stop:I,setMessages:M,input:D,setInput:L,handleInputChange:e=>{L(e.target.value)},handleSubmit:V,isLoading:w,data:x}}}}]);