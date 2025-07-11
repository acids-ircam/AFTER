// inlets & outlets
inlets = 2
outlets = 2
// Global current model
var cur_prior = 0;
var cur_encoder = 0;
var cur_decoder = 0;
var timbre_recv = 0;
var timbre_unp = 0;
var cur_z = 0;
// Instance prefix
var prefix_z = "z";
var prefix_p = "";
// Object creation variable
var p = this.patcher

var cur_connect = []

// Set prefix
function prefix(val_z, val_p)
{
	prefix_z = val_z
	prefix_p = val_p
}

// Create a given model
function create(model_name, prior) 
{
	if (model_name == "off")
	{
		_delete_current()
		return
	}
	else 
	{
		_delete_current()	
	}
	// Replace current model
	if (cur_z != 0)
		_delete_current()

	// Create decoder
	var max_decoder = p.newdefault(100, 350, "nn~", model_name, "generate", 8192);
	max_decoder.rect = [100, 350, 850, 25]
	// Number of latents
	var n_latents = 8//(max_decoder.getboxattr('numinlets') - 2)/2

	// Gather all 
	cur_z = []
	// cur_encoder = max_encoder
	// cur_decoder = max_decoder

	cur_connect.push(max_decoder)

	// Make major in / out connections
	p.connect(p.getnamed("model_props"), 0, max_decoder, 0)
	p.connect(p.getnamed("msg_in"), 0, max_decoder, 0)
	p.connect(max_decoder, 0, p.getnamed("rave_out"), 0)

	var router = p.getnamed("midi_router")
	cur_prior = 0

	var main_plot = p.getnamed("main_plot");

	main_plot.setattr("bkgndpict", model_name + ".png")
	// Create prior

	for (var z = 0; z < n_latents; z+=1)
	{
		// cur_connect = []

		var un_p = p.newdefault(100 + (100 * z), 180, "unpack", "0", "0");
		un_p.rect = [100 + (100 * z), 115, 185 + (100 * z), 115]

		var sig0 = p.newdefault(100 + (100 * z), 140,"sig~");
		sig0.rect = [100 + (100 * z), 150, 140 + (100 * z), 150]

		var sig1= p.newdefault(150 + (100 * z), 140,"sig~");
		sig1.rect = [150 + (100 * z), 150, 190 + (100 * z), 150]

		p.connect(router, z, un_p, 0)
		p.connect(un_p, 0, sig0, 0)
		p.connect(un_p, 1, sig1, 0)
		p.connect(sig0, 0, max_decoder, 2*z)
		p.connect(sig1, 0, max_decoder, 2*z+1)

		cur_connect.push(un_p)
		cur_connect.push(sig0)
		cur_connect.push(sig1)

	}

	var timbre_recv = p.getnamed("timbre_receive")
	var timbre_unp = p.newdefault(100 + (100 * z), 190, "mc.unpack~",8);
	timbre_unp.rect = [100 + (100 * z), 190, 190 + (100 * z), 190]

	p.connect(timbre_recv, 0,timbre_unp,  0)

	for (var j = 0; j < 8; j++)
		{
			p.connect(timbre_unp, j,max_decoder,  n_latents*2 + j)
		}


	cur_connect.push(timbre_unp)


	//### TIMBRE CONNECTION ###
	
	var max_encoder_timbre =  p.newdefault(2150, 920, "nn~", model_name, "timbre", 8192);
	// max_encoder_timbre.rect = [2500, 1200, 400, 25]

	var receive_timbre = p.getnamed("receive_timbre");
	var pack_timbre = p.getnamed("pack_timbre");
	var enable_timbre = p.getnamed("enable_timbre");

	cur_connect.push(max_encoder_timbre)
	
	p.connect(receive_timbre, 0,max_encoder_timbre,  0);
	p.connect(enable_timbre, 0,max_encoder_timbre,  0);


	for (var j = 0; j < 8; j++)
		{
			p.connect(max_encoder_timbre, j ,pack_timbre,  j)
		}


	var sigm2l1 = p.getnamed("sigm2l1");
	var sigm2l2 = p.getnamed("sigm2l2");
	var packm2l = p.getnamed("packm2l");


	var unpackl2m = p.getnamed("unpackl2m");
	var snapl2m1= p.getnamed("snapl2m1");
	var snapl2m2 = p.getnamed("snapl2m2");

	var max_m2l =  p.newdefault(2800, 930, "nn~", model_name, "map2latent");
	var max_l2m =  p.newdefault(3150, 1400, "nn~", model_name, "latent2map");


	p.connect(sigm2l1, 0 ,max_m2l,  0);
	p.connect(sigm2l2, 0 ,max_m2l,  1);

	for (var j = 0; j < 8; j++)
	{
		p.connect(max_m2l, j ,packm2l,  j)
	}
	
	for (var j = 0; j < 8; j++)
		{
			p.connect(unpackl2m, j ,max_l2m,  j)
		}
		
		p.connect(max_l2m, 0, snapl2m1, 0)
		p.connect(max_l2m, 1, snapl2m2, 0)

	cur_connect.push(max_m2l);
	cur_connect.push(max_l2m);

	outlet(1, "create")
	outlet(0, "bang")
}

function delete()
{
	if (cur_z != 0)
		_delete_current()
}

// delete current model
function _delete_current()
{	

	for (var item in cur_connect){
		p.remove(cur_connect[item])
	}

	// p.remove(cur_decoder)
	// p.remove(timbre_recv)
	// p.remove(timbre_unp)

	var main_plot = p.getnamed("main_plot");
	main_plot.setattr("bkgndpict", "background_transparent.png")

	cur_connect = []
	// for (var t in cur_z)
	// {
	// 	var cur_c = cur_z[t]
	// 	for (var o in cur_c)
	// 		p.remove(cur_c[o])
	// }
	// cur_encoder = 0
	// cur_decoder = 0
	// cur_z = 0
}
